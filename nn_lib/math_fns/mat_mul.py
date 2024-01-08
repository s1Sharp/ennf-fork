from typing import Tuple
import numpy as np

from nn_lib.math_fns.function import Function


class MatMul(Function):
    """
    Matrix multiplication function
    """

    def forward(self) -> np.ndarray:
        """
        Multiply two matrices and return their product, matrices are not necessarily 2D, hint:
        https://numpy.org/doc/stable/reference/generated/numpy.matmul.html

        :return: matrix product of the two arguments
        """
        return np.matmul( self.args[0].data , self.args[1].data )

    def _backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients over two matrix multiplication arguments

        :param grad_output: gradient over the result of the multiplication operation
        :return: a tuple of gradients over two multiplication arguments
        """
        # dw = grad_output * (x * y).T * x
        x = self.args[0].data
        y = self.args[1].data
        result_1 = np.matmul( grad_output, y.T )
        result_2 = np.matmul( x.T, grad_output )

        return tuple([ result_1 , result_2 ])
