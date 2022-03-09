from typing import Tuple
import numpy as np

from nn_lib.math_fns.function import Function


class Log(Function):
    """
    Natural logarithm function (e based)
    """

    def forward(self) -> np.ndarray:
        """
        Compute logarithm of the argument, i.e. log(self.args[0].data)

        :return: logarithm of the argument
        """
        return np.log( self.args[0].data )

    def _backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """
        Compute gradient over the logarithm argument

        :param grad_output: gradient over the result of the logarithm function
        :return: a tuple with a single value representing the gradient over the logarithm argument
        """
        return  tuple([ np.multiply(grad_output, np.power(self.args[0].data, -1 )) ])
