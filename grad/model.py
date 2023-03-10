import math

from .tensor import Tensor
from .functional import xavier_uniform, relu, softmax
from .random import uniform

class Module:
    def __init__(self):
        self._param_list = []

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
        xavier_uniform(self.weight)

        if bias:
            bound = 1 / math.sqrt(output_size)
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
        return relu(input)

class Softmax(Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def __call__(self, input):
        return softmax(input, self.dim)

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


