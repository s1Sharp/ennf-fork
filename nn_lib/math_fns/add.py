from typing import Tuple
import numpy as np

from nn_lib.math_fns.function import Function


class Add(Function):
    """
    Addition of two elements
    """

    def forward(self) -> np.ndarray:
        """
        Add two arguments and return their sum

        Note: the result can have different shape because of numpy broadcasting
        https://numpy.org/doc/stable/user/basics.broadcasting.html
        :return: sum of the two arguments
        """
        return np.add( self.args[0].data , self.args[1].data )

    def _backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients over two addition arguments

        Note: because of the broadcasting, arguments and grad_output can have different shapes, we reduce the
        gradients to their original shape inside reduce_gradient() parent method, hence it is ok here for the
        resulting gradients to have shapes different from the original arguments
        :param grad_output: gradient over the result of the addition operation
        :return: a tuple of gradients over two addition arguments
        """
        return tuple( [ grad_output , grad_output ] )
