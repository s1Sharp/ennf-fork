import numpy as np

from typing import TYPE_CHECKING, Tuple
if TYPE_CHECKING:
    from nn_lib import Tensor


class Function:
    """
    Base class for all functions
    """

    def __init__(self, *args: 'Tensor', **kwargs):
        """
        Create a function given the arguments to apply it to. A function has two main methods: forward and backward.
        Forward method computes the result of the function over arguments. Backward function accepts a gradient value
        over the result of the function and returns gradients over the arguments of the function.
        If a function has arguments for which gradients do not need to be computed (e.g. an axis) they should be
        passed inside kwargs.

        :param args: one or multiple arguments for the function; each specific function accepts a fixed number of
            arguments
        :param kwargs: additional key word arguments of the function
        """
        self.args = args
        self.kwargs = kwargs

    def forward(self) -> np.ndarray:
        """
        Forward pass of the function: compute function result based on self.args

        Note: usually the result has the same shape as the arguments, but it might not be the case. For example for
        SumReduce or Slice operations which alter the structure of the tensors, or the operations which are
        broadcastable, e.g. Add, Mul, etc. See: https://numpy.org/doc/stable/user/basics.broadcasting.html
        :return: an np.ndarray representing the result
        """
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Backward pass of the function: from the gradient over the result of the function (grad_output) compute the
        gradient(s) over the arguments of the function. If the function has a single argument, the result should
        nevertheless be a tuple (with length equal to one).

        :param grad_output: gradient over the result of the function
        :return: gradient(s) over the argument(s) of the function that have the same shape(s) as corresponding
            argument(s)
        """
        grads = self._backward(grad_output)
        # The shape of the result can be different from the arguments' shape due to broadcasting. But in order for the
        # .backward() method to be correct the gradient over an argument must have the same shape as an argument. We let
        # the ._backward() method of such functions return gradients that can not have the same shapes as corresponding
        # arguments, but we reduced them properly to the original shape below.
        reduced_grads = tuple(map(lambda it: self.reduce_gradient(it[1], it[0].data.shape), zip(self.args, grads)))
        return reduced_grads

    def _backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Internal backward method of the function that child classes are to override

        :param grad_output: gradient over the result of the function
        :return: gradient(s) over the argument(s) of the function that do not necessarily have the same shape(s) as
            corresponding argument(s)
        """
        raise NotImplementedError

    @staticmethod
    def reduce_gradient(grad: np.ndarray, original_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Reduce broadcasted gradient to the original shape

        :param grad: gradient to reduce
        :param original_shape: the shape that the corresponding argument originally had
        :return:
        """
        if grad.shape == original_shape:
            return grad
        shape_len_diff = len(grad.shape) - len(original_shape)
        expanded_original_shape = (1,) * shape_len_diff + original_shape
        axes_to_reduce_mask = np.array(grad.shape) != np.array(expanded_original_shape)
        axes_to_reduce = tuple(np.nonzero(axes_to_reduce_mask)[0])
        reduced_gradient = np.sum(grad, axis=axes_to_reduce, keepdims=True)
        reduced_gradient = np.squeeze(reduced_gradient, axis=tuple(range(shape_len_diff)))
        assert reduced_gradient.shape == original_shape
        return reduced_gradient
