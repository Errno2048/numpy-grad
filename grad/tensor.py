import numpy as np

class Tensor:
    @classmethod
    def _ensure_tensor(cls, value):
        if isinstance(value, cls):
            return value
        return cls(value, requires_grad=False)

    @classmethod
    def zeros(cls, *shape, requires_grad=False):
        return cls(np.zeros(shape), requires_grad=requires_grad)

    @classmethod
    def zeros_like(cls, tensor, *, requires_grad=False):
        return cls(np.zeros_like(tensor._value), requires_grad=requires_grad)

    @classmethod
    def ones(cls, *shape, requires_grad=False):
        return cls(np.ones(shape), requires_grad=requires_grad)

    @classmethod
    def ones_like(cls, tensor, *, requires_grad=False):
        return cls(np.ones_like(tensor._value), requires_grad=requires_grad)

    @classmethod
    def cat(cls, tensors, dim=0):
        res = np.concatenate([t._value for t in tensors], axis=dim)
        res = cls(res, requires_grad=(sum((t._requires_grad for t in tensors)) > 0))
        res._grad_calculator = CatGrad(tensors, dim)
        return res

    @classmethod
    def stack(cls, tensors, dim=0):
        res = np.stack([t._value for t in tensors], axis=dim)
        res = cls(res, requires_grad=(sum((t._requires_grad for t in tensors)) > 0))
        res._grad_calculator = StackGrad(tensors, dim)
        return res

    def __init__(self, value, requires_grad=False):
        self._value = np.array(value)
        self._grad = np.zeros_like(self._value)
        self._grad_calculator = None
        self._requires_grad = requires_grad

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self._requires_grad = bool(value)

    @property
    def shape(self):
        return self._value.shape

    @property
    def ndim(self):
        return self._value.ndim

    def dim(self):
        return self.ndim

    def size(self, dim):
        return self.shape[dim]

    def _grad_cal(self):
        info_dic = {id(self) : [self, 0]}
        q = [(0, self)]
        topo = []
        # DFS topological sort
        while q:
            op, *args = q.pop()
            if op == 0:
                tensor, *_ = args
                # start traversing a tensor
                id_ = id(tensor)
                info = info_dic.setdefault(id_, [tensor, 0])
                if info[1] == 0:
                    # not visited, set status to uncompleted
                    info[1] = 1
                elif info[1] == 1:
                    raise ValueError('Cyclic graph is not supported.')
                elif info[1] == 2:
                    # already completed
                    continue
                if tensor._grad_calculator is None:
                    # leaf tensor, directly set to completed and ignore it in topo list
                    info[1] = 2
                else:
                    # set to completed after all children are traversed
                    q.append((1, tensor))
                    for child in tensor._grad_calculator.related_tensors():
                        q.append((0, child))
            elif op == 1:
                # set the related tensor to "completed"
                tensor, *_ = args
                info_dic[id(tensor)][1] = 2
                topo.append(tensor)

        for tensor in reversed(topo):
            tensor._grad_calculator.back(tensor)

    def zero_grad(self):
        self._grad = np.zeros_like(self._value)
        if self._requires_grad:
            pass#self._grad_calculator.zero_grad()

    def __str__(self):
        kwargs = {}
        if self._requires_grad and self._grad is not None:
            kwargs['grad'] = str(self._grad)
        return f'Tensor({str(self._value)}{"".join(", " + k + "=" + v for k, v in kwargs.items())})'

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        value = self._value < other._value
        return Tensor(value, requires_grad=False)

    def __le__(self, other):
        value = self._value <= other._value
        return Tensor(value, requires_grad=False)

    def __gt__(self, other):
        value = self._value > other._value
        return Tensor(value, requires_grad=False)

    def __ge__(self, other):
        value = self._value >= other._value
        return Tensor(value, requires_grad=False)

    def __eq__(self, other):
        value = self._value == other._value
        return Tensor(value, requires_grad=False)

    def __ne__(self, other):
        value = self._value != other._value
        return Tensor(value, requires_grad=False)

    def __bool__(self):
        return bool(self._value)

    def __abs__(self):
        res = Tensor(abs(self._value), requires_grad=self._requires_grad)
        res._grad_calculator = AbsGrad(self)
        return res

    def __add__(self, other):
        other = self._ensure_tensor(other)
        res = self._value + other._value
        res = Tensor(res, requires_grad=self._requires_grad or other._requires_grad)
        res._grad_calculator = AddGrad(self, other)
        return res

    def __sub__(self, other):
        other = self._ensure_tensor(other)
        new_other = Tensor(-other._value, requires_grad=other._requires_grad)
        new_other._grad_calculator = OppositeGrad(other)
        return self + new_other

    def __mul__(self, other):
        other = self._ensure_tensor(other)
        res = self._value * other._value
        res = Tensor(res, requires_grad=self._requires_grad or other._requires_grad)
        res._grad_calculator = HadamardGrad(self, other)
        return res

    def __truediv__(self, other):
        other = self._ensure_tensor(other)
        res = self._value / other._value
        res = Tensor(res, requires_grad=self._requires_grad or other._requires_grad)
        res._grad_calculator = DivGrad(self, other)
        return res

    def __matmul__(self, other):
        other = self._ensure_tensor(other)
        res = self._value @ other._value
        res = Tensor(res, requires_grad=self._requires_grad or other._requires_grad)
        res._grad_calculator = MatmulGrad(self, other)
        return res

    def __pow__(self, power, modulo=None):
        other = self._ensure_tensor(power)
        res = self._value ** other._value
        res = Tensor(res, requires_grad=self._requires_grad or other._requires_grad)
        res._grad_calculator = PowGrad(self, other)
        return res

    def __pos__(self):
        res = Tensor(self._value, requires_grad=self._requires_grad)
        res._grad_calculator = IdenticalGrad(self)
        return res

    def __neg__(self):
        res = Tensor(-self._value, requires_grad=self._requires_grad)
        res._grad_calculator = OppositeGrad(self)
        return res

    def __getitem__(self, item):
        res = self._value[item]
        res = Tensor(res, requires_grad=self._requires_grad)
        res._grad_calculator = SelectGrad(self, item)
        return res

    def __setitem__(self, key, value):
        if self._requires_grad:
            raise ValueError('Inplace assignation is not allowed for tensors with gradient. Please use merge() instead.')
        value = self._ensure_tensor(value)
        self._value[key] = value._value
        self._requires_grad = value._requires_grad
        self._grad_calculator = InverseSelectGrad(value, key)

    def merge(self, index, other):
        res = self._value.copy()
        res[index] = other._value
        res = Tensor(res, requires_grad=self._requires_grad or other._requires_grad)
        res._grad_calculator = MergeGrad(self, other, index)
        return res

    def exp(self):
        res = Tensor(np.exp(self._value), requires_grad=self._requires_grad)
        res._grad_calculator = ExpGrad(self)
        return res

    def log(self):
        res = Tensor(np.log(self._value), requires_grad=self._requires_grad)
        res._grad_calculator = LogGrad(self)
        return res

    def sum(self, dim=None, keepdim=False):
        if dim is None:
            value = self._value.sum()
        else:
            value = self._value.sum(dim)
            if keepdim:
                value = np.expand_dims(value, dim)
        res = Tensor(value, requires_grad=self._requires_grad)
        res._grad_calculator = SumGrad(self, dim, keepdim)
        return res

    def mean(self, dim=None, keepdim=False):
        if dim is None:
            value = self._value.mean()
        else:
            value = self._value.mean(dim)
            if keepdim:
                value = np.expand_dims(value, dim)
        res = Tensor(value, requires_grad=self._requires_grad)
        res._grad_calculator = MeanGrad(self, dim, keepdim)
        return res

    def max(self, dim=None, keepdim=False):
        if dim is None:
            value = self._value.max()
        else:
            value = self._value.max(dim)
            if keepdim:
                value = np.expand_dims(value, dim)
        res = Tensor(value, requires_grad=self._requires_grad)
        res._grad_calculator = MaxGrad(self, dim, keepdim)
        return res

    def min(self, dim=None, keepdim=False):
        return -(-self).max(dim=dim, keepdim=keepdim)

    def argmax(self, dim=None, keepdim=False):
        return Tensor(np.argmax(self._value, axis=dim, keepdims=keepdim), requires_grad=False)

    def argmin(self, dim=None, keepdim=False):
        return Tensor(np.argmin(self._value, axis=dim, keepdims=keepdim), requires_grad=False)

    def reshape(self, *shape):
        res = np.reshape(self._value, shape)
        res = Tensor(res, requires_grad=self._requires_grad)
        res._grad_calculator = ReshapeGrad(self)
        return res

    def squeeze(self, dim):
        res = np.squeeze(self._value, dim)
        res = Tensor(res, requires_grad=self._requires_grad)
        res._grad_calculator = ReshapeGrad(self)
        return res

    def unsqueeze(self, dim):
        res = np.expand_dims(self._value, dim)
        res = Tensor(res, requires_grad=self._requires_grad)
        res._grad_calculator = ReshapeGrad(self)
        return res

    def permute(self, *dims):
        if len(dims) == 0:
            dims = tuple(range(-1, -self._value.ndim - 1, -1))
        res = np.transpose(self._value, dims)
        res = Tensor(res, requires_grad=self._requires_grad)
        res._grad_calculator = PermuteGrad(self, dims)
        return res

    def clip(self, min=None, max=None):
        if min is None:
            min = -np.inf
        if max is None:
            max = np.inf
        res = np.clip(self._value, min, max)
        res = Tensor(res, requires_grad=self._requires_grad)
        res._grad_calculator = ClipGrad(self, min, max)
        return res

    def gather(self, dim, index):
        index = self._ensure_tensor(index)
        return self[_gather_indices(self._value, dim, index._value)]

    def gather_merge(self, other, dim, index):
        other = self._ensure_tensor(other)
        index = self._ensure_tensor(index)
        return self.merge(_gather_indices(self._value, dim, index._value), other)

    def to_list(self):
        return self._value.tolist()

    def to_numpy(self):
        return self._value.copy()

    def copy(self):
        res = Tensor(self._value, requires_grad=self._requires_grad)
        res._grad_calculator = IdenticalGrad(self)
        return res

    def detach(self):
        res = Tensor(self._value, requires_grad=False)
        return res

    def backward(self, grad=None):
        assert self._requires_grad, 'The tensor does not have gradient.'
        self._value : np.ndarray
        if grad is None:
            if self._value.size == 1:
                self._grad = np.ones_like(self._value)
            else:
                raise ValueError('Gradient is not provided.')
        else:
            self._grad = grad
        self._grad_cal()

def _gather_indices(value, dim, index):
    dim = dim % value.ndim
    indices = []
    expand = list(range(1, value.ndim))
    for i in range(value.ndim):
        if i == dim:
            indices.append(index)
        else:
            indices.append(np.expand_dims(np.arange(value.shape[i]), expand))
        if i < value.ndim - 1:
            expand[i] = i
    return tuple(indices)

def _shape_list(a : np.ndarray, b : np.ndarray):
    ndim_a, ndim_b = a.ndim, b.ndim
    if ndim_a < ndim_b:
        shapes = [(None, d) for d in b.shape[:-ndim_a]]
        shapes.extend(((da, db) for da, db in zip(a.shape, b.shape[-ndim_a:])))
    else:
        shapes = [(d, None) for d in a.shape[:-ndim_b]]
        shapes.extend(((da, db) for da, db in zip(a.shape[-ndim_b:], b.shape)))
    return list(enumerate(shapes))

def _shape_debroadcast(value : np.ndarray, shape_list, target=0):
    for i, (dim_a, dim_b) in reversed(shape_list):
        if dim_a is None and target == 0 or dim_b is None and target == 1:
            value = np.sum(value, axis=i, keepdims=False)
        elif dim_a == 1 and dim_b != 1 and target == 0 or dim_a != 1 and dim_b == 1 and target == 1:
            value = np.sum(value, axis=i, keepdims=True)
    return value

class Grad:
    def back(self, parent : Tensor):
        pass

    def zero_grad(self):
        for tensor in self.related_tensors():
            if tensor._requires_grad:
                tensor._grad = np.zeros_like(tensor._value)

    def related_tensors(self):
        return ()

class UnaryGrad(Grad):
    def __init__(self, a : Tensor):
        self.a = a

    def related_tensors(self):
        return (self.a,)

class BinaryGrad(Grad):
    def __init__(self, a : Tensor, b : Tensor):
        self.a = a
        self.b = b

    def related_tensors(self):
        return (self.a, self.b)

class MatmulGrad(BinaryGrad):
    def back(self, parent : Tensor):
        if self.a._requires_grad:
            self.a._grad += np.matmul(parent._grad, np.transpose(self.b._value))
        if self.b._requires_grad:
            a = self.a._value.reshape(self.a._value.size // self.a.shape[-1], self.a.shape[-1])
            grad = parent._grad.reshape(parent._grad.size // parent.shape[-1], parent.shape[-1])
            self.b._grad += np.matmul(np.transpose(a), grad)

class AddGrad(BinaryGrad):
    def back(self, parent : Tensor):
        broadcast_shape = _shape_list(self.a._value, self.b._value)
        if self.a._requires_grad:
            self.a._grad += _shape_debroadcast(parent._grad, broadcast_shape, 0)
        if self.b._requires_grad:
            self.b._grad += _shape_debroadcast(parent._grad, broadcast_shape, 1)

class OppositeGrad(UnaryGrad):
    def back(self, parent : Tensor):
        if self.a._requires_grad:
            self.a._grad += -parent._grad

class HadamardGrad(BinaryGrad):
    def back(self, parent : Tensor):
        broadcast_shape = _shape_list(self.a._value, self.b._value)
        if self.a._requires_grad:
            grad = parent._grad * self.b._value
            self.a._grad += _shape_debroadcast(grad, broadcast_shape, 0)
        if self.b._requires_grad:
            grad = parent._grad * self.a._value
            self.b._grad += _shape_debroadcast(grad, broadcast_shape, 1)

class DivGrad(BinaryGrad):
    def back(self, parent : Tensor):
        broadcast_shape = _shape_list(self.a._value, self.b._value)
        if self.a._requires_grad:
            grad = parent._grad / self.b._value
            self.a._grad += _shape_debroadcast(grad, broadcast_shape, 0)
        if self.b._requires_grad:
            grad = parent._grad * (-parent._value / self.b._value)
            self.b._grad += _shape_debroadcast(grad, broadcast_shape, 1)

class ReshapeGrad(UnaryGrad):
    def back(self, parent : Tensor):
        if self.a._requires_grad:
            self.a._grad += np.reshape(parent._grad, self.a._value.shape)

def _inverse_permute(permute):
    res = [None for i in range(len(permute))]
    for i, p in enumerate(permute):
        res[i] = p
    return tuple(res)

class PermuteGrad(UnaryGrad):
    def __init__(self, a, permute):
        super().__init__(a)
        self._permute = permute
        self._inv = _inverse_permute(permute)

    def back(self, parent : Tensor):
        if self.a._requires_grad:
            self.a._grad += np.transpose(parent._grad, self._inv)

class AbsGrad(UnaryGrad):
    def back(self, parent : Tensor):
        if self.a._requires_grad:
            self.a._grad += np.sign(parent._value) * parent._grad

class IdenticalGrad(UnaryGrad):
    def back(self, parent : Tensor):
        if self.a._requires_grad:
            self.a._grad += parent._grad

class PowGrad(BinaryGrad):
    def back(self, parent : Tensor):
        broadcast_shape = _shape_list(self.a._value, self.b._value)
        if self.a._requires_grad:
            grad = parent._grad * self.b._value * parent._value / self.a._value
            self.a._grad += _shape_debroadcast(grad, broadcast_shape, 0)
        if self.b._requires_grad:
            grad = parent._grad * parent._value * np.log(self.a._value)
            self.b._grad += _shape_debroadcast(grad, broadcast_shape, 1)

class LogGrad(UnaryGrad):
    def back(self, parent : Tensor):
        if self.a._requires_grad:
            self.a._grad += parent._grad * 1 / self.a._value

class ExpGrad(UnaryGrad):
    def back(self, parent : Tensor):
        if self.a._requires_grad:
            self.a._grad += parent._grad * parent._value

class AggrGrad(UnaryGrad):
    def __init__(self, a : Tensor, dim, keepdim):
        super().__init__(a)
        self.dim = dim
        self.keepdim = keepdim

class SumGrad(AggrGrad):
    def back(self, parent : Tensor):
        if self.a._requires_grad:
            if self.dim is None:
                self.a._grad += parent._grad * np.ones_like(self.a._value)
            else:
                value = parent._grad
                if not self.keepdim:
                    value = np.expand_dims(value, self.dim)
                self.a._grad += np.repeat(value, self.a._value.shape[self.dim], self.dim)

class MeanGrad(AggrGrad):
    def back(self, parent : Tensor):
        if self.a._requires_grad:
            if self.dim is None:
                self.a._grad += parent._grad * np.ones_like(self.a._value) / self.a._value.size
            else:
                value = parent._grad
                if not self.keepdim:
                    value = np.expand_dims(value, self.dim)
                self.a._grad += np.repeat(value, self.a._value.shape[self.dim], self.dim) / self.a._value.shape[self.dim]

def _permute_to_last_dim(array : np.ndarray, dim):
    permute = list(range(array.ndim))
    permute[dim] = -1
    permute[-1] = dim
    return np.transpose(array, permute)

# TODO:
class MaxGrad(AggrGrad):
    def back(self, parent : Tensor):
        if self.a._requires_grad:
            if self.dim is None:
                mask = np.zeros_like(self.a._value)
                max_index = self.a._value.argmax()
                indices = []
                for shape in reversed(self.a._value.shape):
                    indices.append(max_index % shape)
                    max_index //= shape
                mask[(*reversed(indices),)] = 1
                self.a._grad += parent._grad * mask
            else:
                value = parent._grad
                if not self.keepdim:
                    value = np.expand_dims(value, self.dim)
                a_value = _permute_to_last_dim(self.a._value, self.dim)
                axes = []
                for i in range(a_value.ndim - 1):
                    axis = np.expand_dims(np.arange(a_value.shape[i]), tuple(range(-1, 1 + i - a_value.ndim, -1)))
                    axes.append(axis)
                mask = np.zeros_like(a_value)
                mask[(*axes, np.argmax(a_value, -1))] = 1
                mask = _permute_to_last_dim(mask, self.dim)
                self.a._grad += np.repeat(value, self.a._value.shape[self.dim], self.dim) * mask

class MultiGrad(Grad):
    def __init__(self, tensors, dim=None):
        self.tensors = tensors
        self.dim = dim

    def related_tensors(self):
        return self.tensors

class CatGrad(MultiGrad):
    def __init__(self, tensors, dim):
        super().__init__(tensors, dim)
        self._tensor_pos = []
        total_shape = 0
        for tensor in tensors:
            shape = tensor.shape[dim]
            self._tensor_pos.append((total_shape, total_shape + shape))
            total_shape += shape

    def back(self, parent : Tensor):
        parent_grad = _permute_to_last_dim(parent._grad, self.dim)
        for tensor, (shape_start, shape_end) in zip(self.tensors, self._tensor_pos):
            if tensor._requires_grad:
                grad = _permute_to_last_dim(parent_grad[..., shape_start : shape_end], self.dim)
                tensor._grad += grad

class StackGrad(MultiGrad):
    def back(self, parent : Tensor):
        parent_grad = _permute_to_last_dim(parent._grad, self.dim)
        for index, tensor in enumerate(self.tensors):
            if tensor._requires_grad:
                print(parent_grad.shape, tensor._grad.shape)
                grad = np.expand_dims(parent_grad[..., index], -1)
                grad = _permute_to_last_dim(grad, self.dim)
                tensor._grad += np.reshape(grad, tensor._value.shape)

class SelectGrad(UnaryGrad):
    def __init__(self, a, index):
        super().__init__(a)
        self.index = index

    def back(self, parent : Tensor):
        if self.a._requires_grad:
            delta_grad = np.zeros_like(self.a._value)
            delta_grad[self.index] = parent._grad
            self.a._grad += delta_grad

class InverseSelectGrad(UnaryGrad):
    def __init__(self, a, index):
        super().__init__(a)
        self.index = index

    def back(self, parent : Tensor):
        if self.a._requires_grad:
            self.a._grad += parent._grad[self.index]

class MergeGrad(BinaryGrad):
    def __init__(self, a, b, index):
        super().__init__(a, b)
        self.index = index

    def back(self, parent : Tensor):
        mask = np.zeros_like(self.a._value)
        mask[self.index] = 1
        if self.a._requires_grad:
            self.a._grad += parent._grad * (1 - mask)
        if self.b._requires_grad:
            self.b._grad += parent._grad * mask

class ClipGrad(UnaryGrad):
    def __init__(self, a, min, max):
        super().__init__(a)
        self.clip_min = min
        self.clip_max = max

    def back(self, parent : Tensor):
        if self.a._requires_grad:
            mask = (self.a._value >= self.clip_min) & (self.a._value <= self.clip_max)
            self.a._grad += parent._grad * mask
