from typing import Tuple
import numpy as np

from nn_lib.math_fns.function import Function


class Exp(Function):
    """
    Exponent function
    """

    def forward(self) -> np.ndarray:
        """
        Compute exponent of the argument, i.e. e^(self.args[0].data)

        :return: exponent of the argument
        """
        return np.exp( self.args[0].data )

    def _backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """
        Compute gradient over the exponent argument

        :param grad_output: gradient over the result of the exponent function
        :return: a tuple with a single value representing the gradient over the exponent argument
        """
        return tuple([ np.multiply(grad_output , np.exp(self.args[0].data)) ])