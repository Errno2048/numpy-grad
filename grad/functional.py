import math
import numpy as np

from .tensor import Tensor, UnaryGrad as _UnaryGrad

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

def xavier_uniform(tensor : Tensor, gain : float = 1.0):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    tensor.zero_grad()
    new_value = np.random.uniform(-a, a, size=tensor.shape)
    tensor._value = new_value
    return tensor

def xavier_normal(tensor : Tensor, gain : float = 1.0):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    tensor.zero_grad()
    new_value = np.random.normal(0.0, std, size=tensor.shape)
    tensor._value = new_value
    return tensor

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
            g2 = -parent._value * g1.sum(axis=-1, keepdims=True)
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

def cross_entropy(input, target, dim=-1, *, reduction='mean', target_type='indices'):
    q = softmax(input, dim=dim) + 1e-8
    if target_type == 'onehot':
        pass
    else:
        # indices
        _target = Tensor.zeros_like(input)
        target = _target.gather_merge(1, dim, target.unsqueeze(-1))
    res = -(target * q.log()).sum(dim=dim)
    if reduction == 'sum':
        res = res.sum()
    elif reduction == 'mean':
        res = res.mean()
    else:
        pass
    return res
