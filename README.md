# numpy-grad
 A toy deep learning tool for homework.

```python
from grad.tensor import Tensor
from grad.model import Linear, ReLU, Sequential, Module
from grad.optim import Optimizer
from grad.random import uniform, normal

class Model(Module):
    def __init__():
        super().__init__()
        self.model = Sequential(
            Linear(10, 20),
            ReLU(),
            Linear(20, 5),
        )
        self.register("model")
    
    def __call__(input):
        return self.model(input)

model = Model()
optim = Optimizer(model.parameters(), lr=1e-3)

x = uniform(-1, 1, size=(10, 30, 20, 10))
y = normal(0, 1, size=(10, 30, 20, 5))
pred = model(x)
loss = ((pred - y) ** 2).mean()

optim.zero_grad()
loss.backward()
optim.step()

```

