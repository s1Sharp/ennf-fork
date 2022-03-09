from typing import Tuple, Union
import numpy as np

from nn_lib.math_fns.function import Function


class Slice(Function):
    """
    Slice function
    """

    def __init__(self, *args, slice_obj: Union[int, slice, Tuple[Union[slice, int], ...]]):
        """
        Creates a Slice operation
        Besides the main argument, accepts an object representing a slice

        :param args: value to slice
        :param slice_obj: a python representation of the slice
        """
        super(Slice, self).__init__(*args)
        self.slice_obj = slice_obj

    def forward(self) -> np.ndarray:
        """
        Slice the argument according to the provided slice object, i.e. arg[slice_obj]

        :return: sliced argument
        """
        return self.args[0].data[self.slice_obj]

    def _backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Compute the gradient over argument of the slice operation

        Hint: should be equal to 1 if the corresponding array value is present in the resulting slice and 0 otherwise
        :param grad_output: gradient over the result of the slicing operation
        :return: a tuple with a single value representing the gradient over the slice operation argument
        """
        raise NotImplementedError   # TODO: implement me as an exercise
