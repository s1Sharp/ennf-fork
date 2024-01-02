from typing import Tuple
import numpy as np

from nn_lib.math_fns.function import Function

#https://www.reddit.com/r/MLQuestions/comments/10z6akx/backpropagation_through_a_concatenation_layer/

class Concat2d(Function):
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
        return np.concatenate(self.args[0].data, self.args[1].data, axis=self.kwargs['axis'])

    def _backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """


        Note: because of the broadcasting, arguments and grad_output can have different shapes, we reduce the
        gradients to their original shape inside reduce_gradient() parent method, hence it is ok here for the
        resulting gradients to have shapes different from the original arguments
        :param grad_output: gradient over the result of the addition operation
        :return: a tuple of gradients over two addition arguments
        """
        x = np.ones_like(self.args[0].data)
        y = np.ones_like(self.args[1].data)
        return tuple([grad_output * x, grad_output * y])
