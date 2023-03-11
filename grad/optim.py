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

class SGD(Optimizer):
    def __init__(self, parameters, lr=1e-3, momentum=0.0, dampening=0.0, nesterov=False, weight_decay=0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.momentum_dicts = {}
        self.reset()

    def reset(self):
        super().reset()
        for name, tensor in self._traverse_parameter():
            if tensor.requires_grad:
                self.momentum_dicts[name] = None

    def optimize(self, name, tensor):
        grad = tensor._grad
        grad = grad + self.weight_decay * tensor._value
        if self.momentum != 0.0:
            prev_b = self.momentum_dicts.get(name, None)
            if prev_b is None:
                b = grad
            else:
                b = self.momentum * prev_b + (1 - self.dampening) * grad
            self.momentum_dicts[name] = b
            if self.nesterov:
                grad = grad + self.momentum * b
            else:
                grad = b
        return self.lr * grad

class Adam(Optimizer):
    def __init__(self, parameters, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0, weight_decay_index=2):
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.weight_decay_index = weight_decay_index
        self.momentum_dicts = {}
        self.reset()

    def reset(self):
        super().reset()
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
        res = fix_m / (np.sqrt(fix_v) + self.epsilon)
        if self.weight_decay != 0.0:
            res = res + self.weight_decay * abs(tensor._value ** self.weight_decay_index)
        return self.lr * res
