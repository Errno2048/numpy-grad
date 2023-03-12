import math
from collections.abc import Iterable

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
