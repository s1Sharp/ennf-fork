from typing import List, Tuple

from nn_lib.optim.optimizer import Optimizer
from nn_lib import Tensor
import numpy as np

class Adam(Optimizer):
    """
    Stochastic gradient descent optimizer similar to https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
    """

    def __init__(self, parameters: List[Tensor], lr=1e-3, weight_decay: float = 1e-2,
                 betas: Tuple[float, float] = (0.9, 0.999), eps:float = 1e-8):
        """
        Create an Adam optimizer
        :param parameters: list of parameters of a model
        :param lr: learning rate of the optimizer
        :param weight_decay: a weight decay parameter of the optimizer controlling regularization
        """
        super(Adam, self).__init__(parameters)
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.betas = betas
        self.t = 1
        self.m = []
        self.v = []
        for param in parameters:
            self.m.append(np.zeros_like(param.data))
            self.v.append(np.zeros_like(param.data))


    def step(self) -> None:
        """
        Update parameters of a model by performing a single gradient descent step
        :return: None
        """
        lr = self.lr
        wd = self.weight_decay
        t = self.t

        beta1, beta2 = self.betas
        for i, param in enumerate(self.parameters):
            if param.requires_grad:
                m = self.m[i]
                v = self.v[i]
                grad = param.grad.data  # minimization
                if wd != 0:
                    grad = grad + wd * param.data

                m = beta1 * m + (1 - beta1) * grad
                mt = m / (1 - beta1 ** t)

                v = beta2 * v + (1 - beta2) * (grad ** 2)
                vt = v / (1 - beta2 ** t)

                result = param.data - lr * mt / (np.sqrt(vt) + self.eps)
                param.data = result

                self.m[i] = m
                self.v[i] = v
                self.t = t + 1
        return

    def __call__(self, *args, **kwargs):
        """
        A module can additionally be called as a callable object
        """
        return self.update_param(*args, **kwargs)

    def update_param(self, lr, weight_decay: float = 1e-2,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.betas = betas
    # example
    # optimazer(lr=0.01, weight_decay=5e-3)


