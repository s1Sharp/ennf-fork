from typing import Union, Tuple
import numpy as np

from nn_lib.math_fns.function import Function


class SumReduce(Function):
    """
    Summation reduction over given axis (or axes)
    """

    def __init__(self, *args, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False):
        """
        Create sum reduction function

        :param args: an argument to apply sum for
        :param axis: axis or multiple axes to sum over
        """
        super(SumReduce, self).__init__(*args)

        self.keepdims = keepdims

        if axis is None:
            axis = tuple(range(len(self.args[0].data.shape)))
        if isinstance(axis, int):
            axis = (axis,)
        self.axis = axis

    def forward(self) -> np.ndarray:
        """
        Reduce given axes by summing values over them
        Hint: https://numpy.org/doc/stable/reference/generated/numpy.sum.html

        :return: the reduced value
        """
        return np.sum(self.args[0].data, axis=self.axis, keepdims=self.keepdims)

    def _backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Compute the gradient over the reduction operation
        Hint: values across the reduced axes must have the same gradients

        :param grad_output: the gradient of the result of the reduction
        :return: a tuple with a single value representing the gradient over the reduction argument
        """

        result = np.multiply(np.ones_like(self.args[0].data), grad_output)
        return tuple([result])
