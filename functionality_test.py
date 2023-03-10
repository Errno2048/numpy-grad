from grad import tensor as ts
from grad import functional as F
import numpy as np
import torch

a = np.random.random((3, 5))
b = np.random.random((5, 2))
c = np.random.random((3, 2))
d = np.random.random((3, 2))

ts_a = ts.Tensor(a, requires_grad=True)
ts_b = ts.Tensor(b, requires_grad=True)
ts_c = ts.Tensor(c, requires_grad=True)
ts_d = ts.Tensor(d, requires_grad=True)

t_a = torch.tensor(a, requires_grad=True)
t_b = torch.tensor(b, requires_grad=True)
t_c = torch.tensor(c, requires_grad=True)
t_d = torch.tensor(d, requires_grad=True)

ts_tensor = ((ts.Tensor(np.e) ** (ts_a * ts_a) @ (abs(ts_b) + 1).log()) * ts_c.exp() - ts_d / ts_c).mean()
torch_tensor = (((t_a * t_a).exp() @ (t_b.abs() + 1).log()) * t_c.exp() - t_d / t_c).mean()

ts_tensor.backward()
torch_tensor.backward()

print(t_a.grad.numpy() - ts_a._grad)
print(t_b.grad.numpy() - ts_b._grad)
print(t_c.grad.numpy() - ts_c._grad)
print(t_d.grad.numpy() - ts_d._grad)

a = np.random.random((2, 3, 5))
b = np.random.random((5, 4))
ts_a = ts.Tensor(a, requires_grad=True)
ts_b = ts.Tensor(b, requires_grad=True)
t_a = torch.tensor(a, requires_grad=True)
t_b = torch.tensor(b, requires_grad=True)
ts_tensor = (ts_a @ ts_b).mean()
torch_tensor = (t_a @ t_b).mean()

ts_tensor.backward()
torch_tensor.backward()

print(t_a.grad.numpy() - ts_a._grad)
print(t_b.grad.numpy() - ts_b._grad)

a = np.random.random((2,3,5))
ts_a = ts.Tensor(a, requires_grad=True)
t_a = torch.tensor(a, requires_grad=True)

ts_tensor = F.softmax(ts_a, -1)
ts_tensor_1 = ts_tensor.mean()

torch_tensor = torch.nn.functional.softmax(t_a, -1)
torch_tensor_1 = torch_tensor.mean()

print(torch_tensor_1.detach().numpy() - ts_tensor_1._value)

ts_tensor_1.backward()
torch_tensor_1.backward()

print(t_a.grad.numpy() - ts_a._grad)

detached = ts_tensor.detach()
grad = (ts_tensor - ts_tensor ** 2)
#print(grad._value * ts_tensor._grad - ts_a._grad)

a = np.random.random((10, 10))
b = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1],
              [0, 1, 0], [1, 0, 0], [0, 0, 1],
              [0, 1, 0], [1, 0, 0], [0, 0, 1],
              [0, 1, 0]])
b = np.zeros((10, 10))
b[:, 0] = 1
ts_a = ts.Tensor(a, requires_grad=True)
ts_b = ts.Tensor(b)
t_a = torch.tensor(a, requires_grad=True)
t_b = torch.tensor(b)
ts_tensor = F.cross_entropy(ts_a, ts_b, dim=-1)
torch_tensor = torch.nn.functional.softmax(t_a, dim=-1) + 1e-8
torch_tensor = -(t_b * torch_tensor.log()).mean()

ts_tensor.backward()
torch_tensor.backward()

a = np.random.random((2, 3, 5))
b = np.random.random((5, 4))
ts_a = ts.Tensor(a, requires_grad=True)
ts_b = ts.Tensor(b, requires_grad=True)
t_a = torch.tensor(a, requires_grad=True)
t_b = torch.tensor(b, requires_grad=True)
ts_tensor = (ts_a @ ts_b).max()
torch_tensor = (t_a @ t_b).max()

ts_tensor.backward()
torch_tensor.backward()

print(t_a.grad.numpy() - ts_a._grad)
print(t_b.grad.numpy() - ts_b._grad)

a = np.random.random((20, 30, 50))
b = np.random.random((50, 40))
ts_a = ts.Tensor(a, requires_grad=True)
ts_b = ts.Tensor(b, requires_grad=True)
t_a = torch.tensor(a, requires_grad=True)
t_b = torch.tensor(b, requires_grad=True)
ts_tensor = (ts_a @ ts_b).max(dim=-1).mean()
torch_tensor = (t_a @ t_b).max(dim=-1).values.mean()

ts_tensor.backward()
torch_tensor.backward()

print(t_a.grad.numpy() - ts_a._grad)
print(t_b.grad.numpy() - ts_b._grad)

a = np.random.random((20, 30, 50))
b = np.random.random((50, 40))
ts_a = ts.Tensor(a, requires_grad=True)
ts_b = ts.Tensor(b, requires_grad=True)
t_a = torch.tensor(a, requires_grad=True)
t_b = torch.tensor(b, requires_grad=True)
ts_tensor = (ts_a[:, :, 10:20] @ ts_b[20:30]).max(dim=-1).mean()
torch_tensor = (t_a[:, :, 10:20] @ t_b[20:30]).max(dim=-1).values.mean()

ts_tensor.backward()
torch_tensor.backward()

print(t_a.grad.numpy() - ts_a._grad)
print(t_b.grad.numpy() - ts_b._grad)

a = np.random.random((20, 30, 50))
b = np.random.random((50, 40))
ts_a = ts.Tensor(a, requires_grad=True)
ts_b = ts.Tensor(b, requires_grad=True)
t_a = torch.tensor(a, requires_grad=True)
t_b = torch.tensor(b, requires_grad=True)
ts_a1 = ts.Tensor.cat([ts_a[10:], ts_a[:10]], dim=1)
t_a1 = torch.cat([t_a[10:], t_a[:10]], dim=1)
ts_tensor = (ts_a1[:, :, 10:20] @ ts_b[20:30]).max(dim=-1).mean()
torch_tensor = (t_a1[:, :, 10:20] @ t_b[20:30]).max(dim=-1).values.mean()

ts_tensor.backward()
torch_tensor.backward()

print(t_a.grad.numpy() - ts_a._grad)
print(t_b.grad.numpy() - ts_b._grad)

a = np.random.random((20, 30, 50))
b = np.random.random((50, 40))
ts_a = ts.Tensor(a, requires_grad=True)
ts_b = ts.Tensor(b, requires_grad=True)
t_a = torch.tensor(a, requires_grad=True)
t_b = torch.tensor(b, requires_grad=True)
ts_a1 = ts.Tensor.stack([ts_a[10:], ts_a[:10]], dim=1).reshape(-1, 30, 50).clip(0.3, 0.5)
t_a1 = torch.stack([t_a[10:], t_a[:10]], dim=1).reshape(-1, 30, 50).clip(0.3, 0.5)
print(ts_a1.shape, t_a1.shape)
ts_tensor = (ts_a1[:, :, 10:20] @ ts_b[20:30]).clip(0.1, 0.6).max(dim=-1).mean()
torch_tensor = (t_a1[:, :, 10:20] @ t_b[20:30]).clip(0.1, 0.6).max(dim=-1).values.mean()

ts_tensor.backward()
torch_tensor.backward()

print(t_a.grad.numpy() - ts_a._grad)
print(t_b.grad.numpy() - ts_b._grad)

a = np.random.random((10, 50, 30, 20))
b = np.random.random((50, 40))
ts_a = ts.Tensor(a, requires_grad=True)
ts_b = ts.Tensor(b, requires_grad=True)
t_a = torch.tensor(a, requires_grad=True)
t_b = torch.tensor(b, requires_grad=True)
ts_a0 = ts_a.permute(0, -1, 2, 1)
t_a0 = t_a.permute(0, -1, 2, 1)
ts_a1 = ts_a0.gather(0, ts_a0.argmax(0, keepdim=True)).squeeze(0)
t_a1 = t_a0.gather(0, t_a0.argmax(0, keepdim=True)).squeeze(0)
print(ts_a1.shape, t_a1.shape)
ts_tensor = (ts_a1[:, :, 10:20] @ ts_b[20:30]).clip(0.1, 0.6).max(dim=-1).mean()
torch_tensor = (t_a1[:, :, 10:20] @ t_b[20:30]).clip(0.1, 0.6).max(dim=-1).values.mean()

print(torch_tensor.detach().numpy() - ts_tensor._value)

ts_tensor.backward()
torch_tensor.backward()

print(t_a.grad.numpy() - ts_a._grad)
print(t_b.grad.numpy() - ts_b._grad)

a = np.random.random((2,3,5))
ts_a = ts.Tensor(a, requires_grad=True)
ts_b = ts.Tensor.zeros_like(ts_a)
ts_b1 = ts_b.gather_merge(1, -1, ts_a.argmax(-1, keepdim=True))
print(ts_b1)

a = np.random.random((20, 10))
b = np.zeros((20))
b[0] = 1
ts_a = ts.Tensor(a, requires_grad=True)
ts_b = ts.Tensor(b)
t_a = torch.tensor(a, requires_grad=True)
t_b = torch.tensor(b, dtype=torch.long)
ts_tensor = F.cross_entropy(ts_a, ts_b, dim=-1)
torch_tensor = torch.nn.functional.cross_entropy(t_a, t_b)

ts_tensor.backward()
torch_tensor.backward()
print(torch_tensor)
print(ts_tensor)

print(t_a.grad.numpy() / ts_a._grad)

a = np.random.random((10, 10))
b = np.zeros((10, 10))
b[:, 0] = 1
ts_a = ts.Tensor(a, requires_grad=True)
ts_b = ts.Tensor(b)
t_a = torch.tensor(a, requires_grad=True)
t_b = torch.tensor(b)
ts_tensor = F.cross_entropy(ts_a, ts_b.argmax(dim=-1), dim=-1)
#torch_tensor = torch.nn.functional.softmax(t_a, dim=-1) + 1e-8
#torch_tensor = -(t_b * torch_tensor.log()).mean()
import math
torch_tensor = torch.nn.functional.cross_entropy(t_a, t_b.argmax(dim=-1))

ts_tensor.backward()
torch_tensor.backward()
print(torch_tensor)
print(ts_tensor)

print(t_a.grad.numpy() / ts_a._grad)

a = np.random.random((20, 10))
b = np.zeros((20))
b[0] = 1
ts_a = ts.Tensor(a, requires_grad=True)
ts_b = ts.Tensor(b)
t_a = torch.tensor(a, requires_grad=True)
t_b = torch.tensor(b, dtype=torch.long)
ts_a1 = F.log_softmax(ts_a.tan().sin().cos().arctan(), dim=-1)
ts_tensor = F.nll_loss(ts_a1, ts_b, dim=-1)
torch_tensor = torch.nn.functional.nll_loss(
    torch.nn.functional.log_softmax(
        t_a.tan().sin().cos().arctan(), dim=-1), t_b)

ts_tensor.backward()
torch_tensor.backward()
print(torch_tensor)
print(ts_tensor)

print(t_a.grad.numpy() / ts_a._grad)

a = np.random.random((20, 10))
b = np.zeros((20))
b[0] = 1
ts_a = ts.Tensor(a, requires_grad=True)
ts_b = ts.Tensor(b)
t_a = torch.tensor(a, requires_grad=True)
t_b = torch.tensor(b, dtype=torch.long)
ts_a1 = F.log_softmax(ts_a.arcsin().arcsinh(), dim=-1)
ts_tensor = F.nll_loss(ts_a1, ts_b, dim=-1)
torch_tensor = torch.nn.functional.nll_loss(
    torch.nn.functional.log_softmax(
        t_a.arcsin().arcsinh(), dim=-1), t_b)

ts_tensor.backward()
torch_tensor.backward()
print(torch_tensor)
print(ts_tensor)

print(t_a.grad.numpy() / ts_a._grad)

a = np.random.random((20, 10))
b = np.zeros((20))
b[0] = 1
ts_a = ts.Tensor(a, requires_grad=True)
ts_b = ts.Tensor(b)
t_a = torch.tensor(a, requires_grad=True)
t_b = torch.tensor(b, dtype=torch.long)
ts_a1 = F.log_softmax(((ts_a.arctanh() / 3).arccos() + 2).arccosh(), dim=-1)
ts_tensor = F.nll_loss(ts_a1, ts_b, dim=-1)
torch_tensor = torch.nn.functional.nll_loss(
    torch.nn.functional.log_softmax(
        ((t_a.arctanh() / 3).arccos() + 2).arccosh(), dim=-1), t_b)

ts_tensor.backward()
torch_tensor.backward()
print(torch_tensor)
print(ts_tensor)

print(t_a.grad.numpy() / ts_a._grad)
