from collections import deque
import numpy as np

class Optimizer:
    def __init__(self, parameters, lr=1e-3):
        self.parameters = parameters
        self.lr = lr
        self.step_count = 0

    def _traverse_parameter(self):
        q = deque(self.parameters.items())
        while q:
            k, v = q.popleft()
            if isinstance(v, dict):
                q.extend(((f'{k}.{_k}', _v) for _k, _v in v.items()))
            else:
                yield k, v

    def zero_grad(self):
        for name, tensor in self._traverse_parameter():
            tensor.zero_grad()

    def step(self):
        for name, tensor in self._traverse_parameter():
            if tensor.requires_grad:
                tensor._value = tensor._value - self.optimize(name, tensor)
        self.step_count += 1

    def reset(self):
        self.step_count = 0

    def optimize(self, name, tensor):
        return self.lr * tensor._grad

class Adam(Optimizer):
    def __init__(self, parameters, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.momentum_dicts = {}
        for name, tensor in self._traverse_parameter():
            if tensor.requires_grad:
                self.momentum_dicts[name] = [
                    np.zeros_like(tensor._value), # m
                    np.zeros_like(tensor._value), # v
                ]

    def optimize(self, name, tensor):
        grad = tensor._grad
        m, v = self.momentum_dicts[name]
        t = self.step_count + 1
        new_m = self.beta1 * m + (1 - self.beta1) * grad
        new_v = self.beta2 * v + (1 - self.beta2) * grad * grad
        self.momentum_dicts[name] = [new_m, new_v]
        fix_m = m / (1 - self.beta1 ** t)
        fix_v = v / (1 - self.beta2 ** t)
        return self.lr * fix_m / (np.sqrt(fix_v) + self.epsilon)
