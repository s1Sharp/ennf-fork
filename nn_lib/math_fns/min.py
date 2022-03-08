from typing import Tuple
import numpy as np

from nn_lib.math_fns.function import Function


class Min(Function):
    """
    Minimum over two arrays
    """

    def forward(self) -> np.ndarray:
        """
        Compute minimum over two arrays element-wise, i.e. result[index] =  min(a[index], b[index])

        Note: the result can have different shape because of numpy broadcasting
        https://numpy.org/doc/stable/user/basics.broadcasting.html
        :return: minimum over the two arguments
        """
        return np.minimum( self.args[0].data, self.args[1].data )
        raise NotImplementedError   # TODO: implement me as an exercise

    def _backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients over arguments of the minimum operation
        Important: if two values at some position are equal, the gradient is set to be 0.5

        Note: because of the broadcasting, arguments and grad_output can have different shapes, we reduce the
        gradients to their original shape inside reduce_gradient() parent method, hence it is ok here for the
        resulting gradients to have shapes different from the original arguments
        :param grad_output: gradient over the result of the minimum operation
        :return: a tuple of gradients over arguments of the minimum
        """
        raise NotImplementedError   # TODO: implement me as an exercise
