from typing import List

from nn_lib import Tensor


class Optimizer:
    """
    Base optimizer class
    """
    def __init__(self, parameters: List[Tensor]):
        """
        Create an optimizer from a list of model parameters that it is to optimize
        :param parameters: list of parameters
        """
        self.parameters = parameters
        self.lr = 0

    def step(self) -> None:
        """
        The main method of an optimizer performing a single update step for the parameters
        :return: None
        """
        raise NotImplementedError

    def zero_grad(self):
        """
        Zero the accumulated gradients for the parameters
        :return:
        """
        for param in self.parameters:
            param.zero_grad()
