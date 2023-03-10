from collections import deque

class Optimizer:
    def __init__(self, parameters, lr=1e-3):
        self.parameters = parameters
        self.lr = lr

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
                tensor._value = tensor._value - self.lr * tensor._grad

    def _debug_print_grad(self):
        for name, tensor in self._traverse_parameter():
            print(name, tensor._value, tensor._grad)
