from typing import Tuple
import numpy as np

from nn_lib.math_fns.function import Function


class Sqrt(Function):
    """
    Square root function (e based)
    """

    def forward(self) -> np.ndarray:
        """
        Compute sqrt of the argument, i.e. sqrt(self.args[0].data)

        :return: sqrt of the argument
        """
        return np.power(self.args[0].data, 0.5)

    def _backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """
        Compute gradient over the sqrt argument

        :param grad_output: gradient over the result of the sqrt function
        :return: a tuple with a single value representing the gradient over the sqrt argument
        1 / (2 * sqrt(x))
        """
        return  tuple([ np.multiply(grad_output, 0.5 * np.power(self.args[0].data, 0.5))])
