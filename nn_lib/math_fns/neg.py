from typing import Tuple
import numpy as np

from nn_lib.math_fns.function import Function


class Neg(Function):
    """
    Negation function (additive inverse)
    """

    def forward(self) -> np.ndarray:
        """
        Take negative of the argument, i.e. -self.args[0].data

        :return: negative of the argument
        """
        return np.negative( self.args[0].data)
        raise NotImplementedError   # TODO: implement me as an exercise

    def _backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """
        Compute gradient over the negation argument

        :param grad_output: gradient over the result of the negation
        :return: a tuple with a single value representing the gradient over the negation argument
        """
        raise NotImplementedError   # TODO: implement me as an exercise
