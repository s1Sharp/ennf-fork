from typing import List

from nn_lib.optim.optimizer import Optimizer
from nn_lib import Tensor


class SGD(Optimizer):
    """
    Stochastic gradient descent optimizer similar to https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    """
    def __init__(self, parameters: List[Tensor], lr, weight_decay: float = 5e-4):
        """
        Create an SGD optimizer
        :param parameters: list of parameters of a model
        :param lr: learning rate of the optimizer 
        # TODO: make non-constant (e.g., by providing a callable function)
        :param weight_decay: a weight decay parameter of the optimizer controlling regularization
        """
        super(SGD, self).__init__(parameters)
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self) -> None:
        """
        Update parameters of a model by performing a single gradient descent step
        :return: None
        """
        lr = self.lr
        wd = self.weight_decay
        for param in self.parameters:
            result = (1 - lr * wd) * param.data - lr * param.grad.data
            param.data = result
        return

    def __call__(self, *args, **kwargs):
        """
        A module can additionally be called as a callable object
        """
        return self.update_param(*args, **kwargs)
    
    def update_param(self, lr, weight_decay: float = 5e-4):
        self.lr = lr
        self.weight_decay = weight_decay

    # example
    # optimazer(lr=0.01, weight_decay=5e-3)


