from typing import Tuple
import numpy as np

from nn_lib.math_fns.function import Function


class Mul(Function):
    """
    Multiplication of two elements
    """

    def forward(self) -> np.ndarray:
        """
        Multiply two arguments and return their product

        Note: the result can have different shape because of numpy broadcasting
        https://numpy.org/doc/stable/user/basics.broadcasting.html

        :return: product of the two arguments
        """
        return np.multiply( self.args[0].data, self.args[1].data )

    def _backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients over two multiplication arguments

        Note: because of the broadcasting, arguments and grad_output can have different shapes, we reduce the
        gradients to their original shape inside reduce_gradient() parent method, hence it is ok here for the
        resulting gradients to have shapes different from the original arguments
        :param grad_output: gradient over the result of the multiplication operation
        :return: a tuple of gradients over two multiplication arguments
        """
        return tuple( [grad_output * self.args[1].data , grad_output * self.args[0].data] )
