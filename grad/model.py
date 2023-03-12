import math
from collections.abc import Iterable, Sized

from .tensor import Tensor
from .random import uniform
from . import functional as F

class Module:
    def __init__(self):
        self._param_list = []
        self._eval_mode = False

    @property
    def eval_mode(self):
        return self._eval_mode

    def eval(self):
        self._eval_mode = True

    def train(self):
        self._eval_mode = False

    def register(self, name, ref=None):
        self._param_list.append((name, ref))

    def parameters(self):
        res = {}
        for param, ref in self._param_list:
            if ref is None:
                ref = getattr(self, param)
            if isinstance(ref, Module):
                res[param] = ref.parameters()
            elif isinstance(ref, Tensor):
                res[param] = ref
        return res

class Linear(Module):
    def __init__(self, input_size, output_size, bias=True):
        super().__init__()
        self.weight = Tensor.zeros(input_size, output_size, requires_grad=True)
        self.register('weight')
        F.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            fan_in, _  = F._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias = uniform(-bound, bound, (output_size, ), requires_grad=True)
            self.register('bias')
        else:
            self.bias = None

    def __call__(self, input):
        res = input @ self.weight
        if self.bias is not None:
            res = res + self.bias
        return res

class ReLU(Module):
    def __call__(self, input):
        return F.relu(input)

class Softmax(Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def __call__(self, input):
        return F.softmax(input, self.dim)

class Sequential(Module):
    def __init__(self, *models):
        super().__init__()
        self.models = models
        for i, model in enumerate(models):
            self.register(f'model{i}', model)

    def __call__(self, input):
        for model in self.models:
            input = model(input)
        return input

class Norm(Module):
    def __init__(self, var_shape=None, affine_shape=None, eps=1e-5, momentum=0.1):
        super().__init__()
        self._affine = affine_shape is not None
        self._eps = eps
        if affine_shape is not None:
            self.weight = Tensor.ones(*affine_shape, requires_grad=True)
            self.bias = Tensor.zeros(*affine_shape, requires_grad=True)
            self.register('weight')
            self.register('bias')
        else:
            self.weight = None
            self.bias = None
        self._momentum = momentum
        self._track_running_stats = var_shape is not None
        if self._track_running_stats:
            self._mean = Tensor.zeros(*var_shape)
            self._var = Tensor.zeros(*var_shape)
            self._num_steps = 0
        else:
            self._num_steps = None
            self._mean = None
            self._var = None

    def _norm_axes(self):
        # is inclusion, dims
        return True, (0,)

    def _norm_shape_lists(self, tensor : Tensor):
        is_inclusion, norm_axes = self._norm_axes()
        if is_inclusion:
            is_norm = [False for i in range(tensor.ndim)]
            for i in norm_axes:
                is_norm[i] = True
            no_norm_axes = []
            no_norm_shapes = []
            for i in range(tensor.ndim):
                if not is_norm[i]:
                    no_norm_axes.append(i)
                    no_norm_shapes.append(tensor.shape[i])
        else:
            # exclusion
            no_norm_axes = norm_axes
            is_norm = [True for i in range(tensor.ndim)]
            for i in no_norm_axes:
                is_norm[i] = False
            norm_axes = []
            for i in range(tensor.ndim):
                if is_norm[i]:
                    norm_axes.append(i)
            no_norm_shapes = (tensor.shape[dim] for dim in no_norm_axes)
        return norm_axes, no_norm_axes, no_norm_shapes

    def _norm_permute(self, tensor : Tensor):
        norm_axes, no_norm_axes, no_norm_shapes = self._norm_shape_lists(tensor)
        return tensor.permute(*no_norm_axes, *norm_axes).reshape(*no_norm_shapes, -1)

    def __call__(self, input : Tensor):
        # target: dim 1
        sample = self._norm_permute(input)
        sample_mean = sample.mean(dim=-1, keepdim=False)
        sample_var = sample.var(dim=-1, unbiased=False, keepdim=False)
        n = sample.shape[-1]
        assert n > 1, 'cannot do normalization with only one sample'

        if self._track_running_stats:
            if self.eval_mode:
                estimate_mean = self._mean
                estimate_var = self._var
            else:
                estimate_mean = (1 - self._momentum) * self._mean + self._momentum * sample_mean.detach()
                estimate_var = (1 - self._momentum) * self._var + self._momentum * sample_var.detach()
                self._mean = estimate_mean
                self._var = estimate_var
        else:
            estimate_mean = sample_mean
            estimate_var = sample_var

        estimate_var = estimate_var * n / (n - 1)

        _, no_norm_axes, *_ = self._norm_shape_lists(input)
        shape = [1 for i in range(input.ndim)]
        for dim in no_norm_axes:
            shape[dim] = input.shape[dim]
        estimate_mean = estimate_mean.reshape(*shape)
        estimate_var = estimate_var.reshape(*shape)
        res = (input - estimate_mean) / (estimate_var + self._eps).sqrt()
        if self._affine:
            res = res * self.weight + self.bias

        return res

class BatchNorm(Norm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        if track_running_stats:
            var_shape = (num_features, )
        else:
            var_shape = None
        if affine:
            affine_shape = (1, )
        else:
            affine_shape = None
        super().__init__(var_shape, affine_shape, eps, momentum)

    def _norm_axes(self):
        return False, (1, )

class LayerNorm(Norm):
    def __init__(self, shapes, eps=1e-5, affine=True):
        if not isinstance(shapes, Iterable):
            shapes = (shapes, )
        if affine:
            affine_shape = shapes
        else:
            affine_shape = None
        super().__init__(None, affine_shape, eps, 0.1)
        self._norm_shapes = shapes
        self._norm_dims = tuple(range(-len(shapes), 0, 1))

    def _norm_axes(self):
        return True, self._norm_dims

def _n_repeat(value, n):
    if isinstance(n, Iterable):
        return n
    return tuple(value for i in range(n))

def _check_size(value, n):
    if isinstance(value, Iterable):
        assert isinstance(value, Sized) and len(value) == n, f'value {value} incompatible with size {n}'

class Conv(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        self.kernel = Tensor.zeros(out_channels, in_channels // groups, *kernel_size, requires_grad=True)
        self.register('kernel')
        F.kaiming_uniform_(self.kernel, a=math.sqrt(5))
        if bias:
            fan_in, _ = F._calculate_fan_in_and_fan_out(self.kernel)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            self.bias = uniform(-bound, bound, (out_channels, ), requires_grad=True)
            self.register('bias')
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def __call__(self, input):
        return F.conv(input, self.kernel, padding=self.padding, stride=self.stride, dilation=self.dilation)

class _ConvNd(Conv):
    def __init__(
        self,
        n,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        for v in (kernel_size, stride, padding, dilation):
            _check_size(v, n)
        super().__init__(
            in_channels,
            out_channels,
            _n_repeat(kernel_size, n),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self._n = n

    def __call__(self, input):
        assert input.ndim in (self._n + 1, self._n + 2), f'input shape {input.shape} incompatible with {self._n}'
        return super().__call__(input)

class Conv1d(_ConvNd):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__(
            1, in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
        )

class Conv2d(_ConvNd):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__(
            2, in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
        )

class Conv3d(_ConvNd):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__(
            3, in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
        )

# RNNs are from pytorch
class _RNNCellBase(Module):
    def __init__(self, input_size, hidden_size, bias=True, num_chunks=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        self.weight_ih = uniform(-stdv, stdv, (input_size, num_chunks * hidden_size), requires_grad=True)
        self.weight_hh = uniform(-stdv, stdv, (hidden_size, num_chunks * hidden_size), requires_grad=True)
        self.register('weight_ih')
        self.register('weight_hh')
        if bias:
            self.bias_ih = uniform(-stdv, stdv, (num_chunks * hidden_size, ), requires_grad=True)
            self.bias_hh = uniform(-stdv, stdv, (num_chunks * hidden_size, ), requires_grad=True)
            self.register('bias_ih')
            self.register('bias_hh')
        else:
            self.bias_ih = None
            self.bias_hh = None

def _rnn_base_cal(input : Tensor, hx, weight_ih, weight_hh, bias_ih, bias_hh):
    res = input @ weight_ih + hx @ weight_hh
    if bias_ih is not None:
        res = res + bias_ih
    if bias_hh is not None:
        res = res + bias_hh
    return res

class RNNCell(_RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        super().__init__(input_size, hidden_size, bias=bias, num_chunks=1)
        self.nonlinearity = nonlinearity

    def __call__(self, input : Tensor, hx = None):
        assert input.dim() in (1, 2), \
            f"RNNCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
        is_batched = input.ndim == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = Tensor.zeros(input.shape[0], self.hidden_size)
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx

        res = _rnn_base_cal(input, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)

        if self.nonlinearity == "tanh":
            res = res.tanh()
        elif self.nonlinearity == "relu":
            res = F.relu(res)
        else:
            raise RuntimeError(
                "Unknown nonlinearity: {}".format(self.nonlinearity))

        if not is_batched:
            res = res.squeeze(0)

        return res

class LSTMCell(_RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias=bias, num_chunks=4)

    def forward(self, input : Tensor, hx=None):
        assert input.dim() in (1, 2), \
            f"LSTMCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            zeros = Tensor.zeros(input.size(0), self.hidden_size)
            hx = (zeros, zeros)
        else:
            hx = (hx[0].unsqueeze(0), hx[1].unsqueeze(0)) if not is_batched else hx

        hx, cx = hx

        res = _rnn_base_cal(input, hx, self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
        res = res.reshape(*res.shape[:-1], res.shape[-1] // 4, 4)
        i, f, g, o = res[..., 0], res[..., 1], res[..., 2], res[..., 3]
        i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
        c = f * cx + i * g
        h = o * c.tanh()
        res = (h, c)

        if not is_batched:
            res = (res[0].squeeze(0), res[1].squeeze(0))
        return res

class GRUCell(_RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__(input_size, hidden_size, bias=bias, num_chunks=3)

    def forward(self, input : Tensor, hx = None):
        assert input.dim() in (1, 2), \
            f"GRUCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = Tensor.zeros(input.size(0), self.hidden_size)
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx

        weight_ih = self.weight_ih.reshape(*self.weight_ih.shape[:-1], self.weight_ih.shape[-1] // 3, 3)
        weight_hh = self.weight_hh.reshape(*self.weight_ih.shape[:-1], self.weight_ih.shape[-1] // 3, 3)
        w_ir, w_iz, w_in = weight_ih[..., 0], weight_ih[..., 1], weight_ih[..., 2]
        w_hr, w_hz, w_hn = weight_hh[..., 0], weight_hh[..., 1], weight_hh[..., 2]
        if self.bias_ih:
            bias_ih = self.bias_ih.reshape(*self.bias_ih.shape[:-1], self.bias_ih.shape[-1] // 3, 3)
            b_ir, b_iz, b_in = bias_ih[..., 0], bias_ih[..., 1], bias_ih[..., 2]
        else:
            b_ir, b_iz, b_in = None, None, None
        if self.bias_hh:
            bias_hh = self.bias_hh.reshape(*self.bias_hh.shape[:-1], self.bias_hh.shape[-1] // 3, 3)
            b_hr, b_hz, b_hn = bias_hh[..., 0], bias_hh[..., 1], bias_hh[..., 2]
        else:
            b_hr, b_hz, b_hn = None, None, None

        r = _rnn_base_cal(input, hx, w_ir, w_hr, b_ir, b_hr).sigmoid()
        z = _rnn_base_cal(input, hx, w_iz, w_hz, b_iz, b_hz).sigmoid()
        if b_hn is not None:
            _v = r * (hx @ w_hn)
        else:
            _v = r * (hx @ w_hn + b_hn)
        n = input @ w_in + _v
        if b_in is not None:
            n = n + b_in
        n = n.tanh()

        res = (1 - z) * n + z * hx

        if not is_batched:
            res = res.squeeze(0)

        return res

class _RNNBase(Module):
    def __init__(
        self,
        unit : type,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0.0,
        bidirectional=False,
        *args, **kwargs,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.dropout = dropout

        unit_gen = lambda: unit(input_size, hidden_size, bias, *args, **kwargs)
        bi_unit_gen = lambda: (unit_gen(), unit_gen()) if bidirectional else unit_gen()
        self.units = [bi_unit_gen() for i in range(num_layers)]
        for i, unit in enumerate(self.units):
            if bidirectional:
                unit1, unit2 = unit
                self.register(f'unit{i}_1', unit1)
                self.register(f'unit{i}_2', unit2)
            else:
                self.register(f'unit{i}', unit)

    def __call__(self, input, hx = None):
        # TODO:
        pass
