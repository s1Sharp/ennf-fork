from typing import Tuple
import numpy as np

from nn_lib.math_fns.function import Function


class Pow2(Function):
    """
    pow2 root function (e based)
    """

    def forward(self) -> np.ndarray:
        """
        Compute pow2 of the argument, i.e. pow2(self.args[0].data)

        :return: pow2 of the argument
        """
        return np.power(self.args[0].data, 2)

    def _backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """
        Compute gradient over the sqrt argument

        :param grad_output: gradient over the result of the pow2 function
        :return: a tuple with a single value representing the gradient over the pow2 argument
        2 * x
        """
        return  tuple([ np.multiply(grad_output, 2 * self.args[0].data)])
