from .tensor import Tensor
import numpy as np

def uniform(a, b, size=None, *, requires_grad=False):
    return Tensor(np.random.uniform(a, b, size), requires_grad=requires_grad)

def normal(mean, std, size=None, *, requires_grad=False):
    return Tensor(np.random.normal(mean, std, size), requires_grad=requires_grad)
