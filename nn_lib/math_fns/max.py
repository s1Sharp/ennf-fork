from typing import Tuple
import numpy as np

from nn_lib.math_fns.function import Function


class Max(Function):
    """
    Maximum over two arrays
    """

    def forward(self) -> np.ndarray:
        """
        Compute maximum over two arrays element-wise, i.e. result[index] =  max(a[index], b[index])

        Note: the result can have different shape because of numpy broadcasting
        https://numpy.org/doc/stable/user/basics.broadcasting.html
        :return: maximum over the two arguments
        """
        return np.maximum( self.args[0].data, self.args[1].data )

    def _backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients over arguments of the maximum operation
        Important: if two values at some position are equal, the gradient is set to be 0.5

        Note: because of the broadcasting, arguments and grad_output can have different shapes, we reduce the
        gradients to their original shape inside reduce_gradient() parent method, hence it is ok here for the
        resulting gradients to have shapes different from the original arguments
        :param grad_output: gradient over the result of the maximum operation
        :return: a tuple of gradients over arguments of the maximum
        """
        x = self.args[0].data
        y = self.args[1].data

        #get dQ/dx = dQ/dt*dt/dx

        #==> dQ/dt
        result_1 = np.where( x > y, 1, 0 )
        result_1 = np.where( x == y, 0.5, result_1 )
        result_2 = np.where( y > x, 1, 0 )
        result_2 = np.where( y == x, 0.5, result_2 )

        #==> *dt/dx
        result_1 = np.multiply(result_1, grad_output)
        result_2 = np.multiply(result_2, grad_output)

        return tuple([ result_1, result_2 ])

