import math
import numpy as np

class Scheduler:
    def __init__(self, optim):
        self.optim = optim
        self.step_count = 0
        self.init_lr = None

    def reset(self):
        self.step_count = 0
        if self.init_lr is not None:
            self.optim.lr = self.init_lr

    def _adjust_lr(self):
        return self.init_lr

    def step(self, epoch=None):
        if self.init_lr is None:
            self.init_lr = self.optim.lr
        if epoch is not None:
            self.step_count = epoch
        else:
            self.step_count += 1
        self.optim.lr = self._adjust_lr()

class StepLR(Scheduler):
    def __init__(self, optim, step_size, gamma=0.1):
        super().__init__(optim)
        self.step_size = step_size
        self.gamma = gamma

    def _adjust_lr(self):
        decay = self.step_count // self.step_size
        return self.init_lr * self.gamma ** decay

class MultiStepLR(Scheduler):
    def __init__(self, optim, steps : list, gamma=0.1):
        super().__init__(optim)
        self.steps = np.array(steps)
        self.gamma = gamma

    def _adjust_lr(self):
        decay = (self.steps <= self.step_count).sum()
        return self.init_lr * self.gamma ** decay

class CosineAnnealingLR(Scheduler):
    def __init__(self, optim, T_max, eta_min=0):
        super().__init__(optim)
        self.T_max = T_max
        self.eta_min = eta_min

    def _adjust_lr(self):
        cos = math.cos(self.step_count / self.T_max * math.pi)
        return self.eta_min + (self.init_lr - self.eta_min) * (1 + cos) / 2
