import math
import numpy as np
from collections.abc import Iterable

from .tensor import Tensor, UnaryGrad as _UnaryGrad, BinaryGrad as _BinaryGrad
from . import conv as _conv

# these initialization methods are from pytorch
def _calculate_fan_in_and_fan_out(tensor : Tensor):
    assert tensor.ndim >= 2, 'Fan in and fan out can not be computed for tensor with fewer than 2 dimensions'
    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if tensor.ndim > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def uniform_(tensor : Tensor, low, high):
    tensor.zero_grad()
    new_value = np.random.uniform(low, high, size=tensor.shape)
    tensor._value = new_value
    return tensor

def normal_(tensor : Tensor, mean, std):
    tensor.zero_grad()
    new_value = np.random.normal(mean, std, size=tensor.shape)
    tensor._value = new_value
    return tensor

def xavier_uniform_(tensor : Tensor, gain : float = 1.0):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    tensor.zero_grad()
    new_value = np.random.uniform(-a, a, size=tensor.shape)
    tensor._value = new_value
    return tensor

def xavier_normal_(tensor : Tensor, gain : float = 1.0):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    tensor.zero_grad()
    new_value = np.random.normal(0.0, std, size=tensor.shape)
    tensor._value = new_value
    return tensor

def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out

def calculate_gain(nonlinearity, param=None):
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))

def kaiming_uniform_(
    tensor: Tensor, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu'
):
    if 0 in tensor.shape:
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return uniform_(tensor, -bound, bound)

class _ReLUGrad(_UnaryGrad):
    def back(self, parent : Tensor):
        if self.a._requires_grad:
            self.a._grad += parent._grad * (self.a._value >= 0)

def relu(tensor : Tensor):
    res = Tensor(tensor._value * (tensor._value >= 0), requires_grad=tensor._requires_grad)
    res._grad_calculator = _ReLUGrad(tensor)
    return res

class _SoftmaxGrad(_UnaryGrad):
    def __init__(self, a, dim=-1):
        super().__init__(a)
        self.dim = dim

    def back(self, parent : Tensor):
        if self.a._requires_grad:
            g1 = parent._grad * parent._value
            g2 = -parent._value * g1.sum(axis=self.dim, keepdims=True)
            self.a._grad += g1 + g2

def softmax(tensor : Tensor, dim=-1):
    res = tensor._value
    res_max = np.max(res, axis=dim, keepdims=True)
    res = np.exp(res - res_max)
    sum = np.sum(res, axis=dim, keepdims=True)
    res = res / sum
    res = Tensor(res, requires_grad=tensor._requires_grad)
    res._grad_calculator = _SoftmaxGrad(tensor, dim=dim)
    return res

class _LogSoftmaxGrad(_UnaryGrad):
    def __init__(self, a, dim=-1):
        super().__init__(a)
        self.dim = dim

    def back(self, parent : Tensor):
        if self.a._requires_grad:
            self.a._grad += parent._grad - np.exp(parent._value) * parent._grad.sum(axis=self.dim, keepdims=True)

def log_softmax(tensor : Tensor, dim=-1):
    res = tensor._value
    res_max = np.max(res, axis=dim, keepdims=True)
    res1 = res - res_max
    res2 = np.sum(np.exp(res1), axis=dim, keepdims=True)
    res = res1 - np.log(res2)
    res = Tensor(res, requires_grad=tensor._requires_grad)
    res._grad_calculator = _LogSoftmaxGrad(tensor, dim=dim)
    return res

def _reduce(input, reduction):
    if reduction == 'sum':
        res = input.sum()
    elif reduction == 'mean':
        res = input.mean()
    elif reduction == 'batchmean':
        res = input.mean() / input.shape[0]
    else:
        res = input
    return res

def cross_entropy(input, target, dim=-1, *, reduction='mean', target_type='indices'):
    q = softmax(input, dim=dim) + 1e-8
    if target_type == 'onehot':
        pass
    else:
        # indices
        _target = Tensor.zeros_like(input)
        target = _target.gather_merge(1, dim, target.unsqueeze(-1))
    res = -(target * q.log()).sum(dim=dim)
    return _reduce(res, reduction)

def kl_div(input, target, reduction='mean', log_target=False, log_input=True):
    if not log_input:
        input = input.log()
    if log_target:
        res = target.exp() * (target - input)
    else:
        res = target * (target.log() - input)
    return _reduce(res, reduction)

def dropout(input, p=0.5):
    assert 0.0 <= p < 1.0, 'Invalid dropout probability'
    mask = Tensor((np.random.random(input.shape) >= p).astype(np.float_))
    return input * mask / (1 - p)

def nll_loss(input, target, dim=-1, weight=None, *, reduction='mean', target_type='indices'):
    if target_type == 'onehot':
        pass
    else:
        # indices
        _target = Tensor.zeros_like(input)
        target = _target.gather_merge(1, dim, target.unsqueeze(-1))
    if weight is not None:
        reshape = [1 for i in range(target.ndim)]
        reshape[dim] = target.shape[dim]
        weight = weight.reshape(*reshape)
        target = target * weight
    res = (-target * input).sum(dim=dim, keepdim=False)
    return _reduce(res, reduction)

def ln_loss(input, target, n, reduction='mean'):
    return _reduce(((input - target) ** n).abs(), reduction)

def l1_loss(input, target, reduction='mean'):
    return ln_loss(input, target, 1, reduction)

def mse_loss(input, target, reduction='mean'):
    return ln_loss(input, target, 2, reduction)

class _PadGrad(_UnaryGrad):
    def __init__(self, a, padding):
        super().__init__(a)
        self.padding = padding

    def back(self, parent : Tensor):
        if self.a._requires_grad:
            indices = []
            for p, shape in zip(self.padding, self.a._value.shape):
                if isinstance(p, Iterable):
                    p, *_ = p
                indices.append(slice(p, p + shape))
            self.a._grad += parent._grad[tuple(indices)]

def zero_pad(input, padding, ignore_dims=2):
    res, padding = _conv._np_padding(input._value, padding, ignore_dims=ignore_dims)
    res = Tensor(res, requires_grad=input._requires_grad)
    res._grad_calculator = _PadGrad(input, padding)
    return res

def _np_transpose(array, a, b):
    permute = list(range(array.ndim))
    permute[a], permute[b] = b, a
    return np.transpose(array, permute)

class _ConvGrad(_BinaryGrad):
    def __init__(self, input, kernel, stride : tuple, dilation : tuple):
        super().__init__(input, kernel)
        self.stride = stride
        self.dilation = dilation

    def back(self, parent : Tensor):
        input, kernel = self.a._value, self.b._value
        grad = parent._grad
        batch_expanded = input.ndim < kernel.ndim
        if batch_expanded:
            input = np.expand_dims(input, 0)
            grad = np.expand_dims(input, 0)
        batch_size = input.shape[0]
        in_channel, out_channel = input.shape[1], kernel.shape[0]
        # in_channel / group
        kernel_in_channel = kernel.shape[1]
        group = in_channel // kernel_in_channel
        kernel_out_channel = out_channel // group

        if self.a._requires_grad:
            dilated_kernel = np.zeros((*kernel.shape[:2], *(1 + d * (k - 1) for d, k in zip(self.dilation, kernel.shape[2:]))))
            dilated_indices = tuple(slice(None, None, d) for d in self.dilation)
            dilated_kernel[(..., *dilated_indices)] = kernel
            rev_indices = tuple(slice(None, None, -1) for s in grad.shape[2:])
            rev_grad = grad[(..., *rev_indices)]
            kernel_padding = tuple((0, 0, *(s * (g - 1) for s, g in zip(self.stride, grad.shape[2:]))))
            # (out, in / group, *size)
            new_kernel, *_ = _conv._np_padding(dilated_kernel, padding=kernel_padding)

            # (batch_size * group, kernel_out_channel, kernel_in_channel, *size)
            new_grad = np.reshape(rev_grad, (batch_size * group, kernel_out_channel, 1, *rev_grad.shape[2:]))
            new_grad = np.tile(new_grad, (1, 1, kernel_in_channel, *(1 for _ in new_grad.shape[3:])))
            # (batch_size * group, kernel_out_channel, kernel_in_channel, *size)
            new_kernel = np.tile(new_kernel, (batch_size, *(1 for _ in new_kernel.shape[1:])))
            new_kernel = np.reshape(new_kernel, (batch_size * group, kernel_out_channel, kernel_in_channel, *new_kernel.shape[2:]))

            # (batch_size * group * kernel_in_channel, 1, kernel_out_channel, *size)
            new_grad = _np_transpose(new_grad, 1, 2)
            new_grad = np.expand_dims(new_grad, 2)
            new_grad = np.reshape(new_grad, (-1, 1, kernel_out_channel, *new_grad.shape[4:]))
            # (batch_size * group * kernel_in_channel, kernel_out_channel, *size)
            new_kernel = _np_transpose(new_kernel, 1, 2)
            new_kernel = np.reshape(new_kernel, (-1, kernel_out_channel, *new_kernel.shape[3:]))

            # (batch_size * group * kernel_in_channel, 1, *size)
            input_grad = _conv._np_conv(
                new_kernel, new_grad,
                stride=1, dilation=self.stride,
                keep_in_channel=False, batch_kernel=True,
            )
            input_grad = np.reshape(input_grad, (batch_size, in_channel, *input_grad.shape[2:]))
            if batch_expanded:
                input_grad = np.squeeze(input_grad, axis=0)
            self.a._grad += input_grad

        if self.b._requires_grad:
            new_input = np.reshape(input, (batch_size * group, kernel_in_channel, *input.shape[2:]))
            new_grad = np.reshape(grad, (batch_size * group, kernel_out_channel, 1, *grad.shape[2:]))
            new_grad = np.tile(new_grad, (1, 1, kernel_in_channel, *(1 for _ in new_grad.shape[3:])))
            # (N * group, kernel_out_channel, kernel_in_channel, *size)
            kernel_grad = _conv._np_conv(
                new_input, new_grad,
                stride=self.dilation, dilation=self.stride,
                keep_in_channel=True, batch_kernel=True,
            )
            kernel_grad = np.reshape(kernel_grad, (batch_size, group * kernel_out_channel, kernel_in_channel, *kernel.shape[2:]))
            if batch_expanded:
                kernel_grad = np.squeeze(kernel_grad, axis=0)
            else:
                kernel_grad = kernel_grad.sum(axis=0)
            self.b._grad += kernel_grad

def conv(input, kernel, padding=0, stride=1, dilation=1):
    batched = input.ndim - kernel.ndim
    assert batched in (-1, 0), f'size mismatch for input ({input.shape}) and kernel ({kernel.shape})'
    input = zero_pad(input, padding, ignore_dims=2 + batched)

    v_input = input._value
    v_kernel = kernel._value
    res, _, stride, dilation = _conv._np_conv(v_input, v_kernel, stride=stride, dilation=dilation, detail=True)
    res = Tensor(res, requires_grad=input._requires_grad or kernel._requires_grad)
    res._grad_calculator = _ConvGrad(input, kernel, stride, dilation)
    return res
