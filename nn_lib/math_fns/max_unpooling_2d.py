from typing import Tuple
import numpy as np

from nn_lib.math_fns.function import Function
from numba import jit
#https://stackoverflow.com/questions/38576151/what-is-happening-with-max-pool-backward-in-tensorflow
#https://tinynet.autoai.org/en/latest/induction/unpooling.html
#https://arxiv.org/pdf/2210.10922.pdf

class MaxUnpool2d(Function):
    """
    max unpooling of activation map
    """

    def __init__(self, *args: 'Tensor', **kwargs):
        super().__init__(*args,**kwargs)

    #@jit
    def forward(self) -> np.ndarray:
        """
        Add two arguments and return their sum

        Note: the result can have different shape because of numpy broadcasting
        https://numpy.org/doc/stable/user/basics.broadcasting.html
        :return: sum of the two arguments

        !!! NO wrt to sliding window and kernel size!
        """
        unpooling_indicies = self.args[1].data
        stride = self.kwargs['stride']
        sliding_window_size = self.kwargs['sliding_window_size']
        assert (stride==2 and sliding_window_size == 2) # to fix that recalculate np.ones(2,2) on 32
        result = np.zeros_like(self.args[1].data)
        x = self.args[0].data
        shape = x.shape
        for i in range(shape[0]):
            for j in range(shape[-1]):
                result[i,...,j] = np.kron(x[i,...,j], np.ones((2,2), dtype=x.dtype))
        result = result * unpooling_indicies
        return result

    #@jit
    def _backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray]:
        """
        Note: because of the broadcasting, arguments and grad_output can have different shapes, we reduce the
        gradients to their original shape inside reduce_gradient() parent method, hence it is ok here for the
        resulting gradients to have shapes different from the original arguments
        :param grad_output: gradient over the result of the addition operation
        :return: a tuple of gradients over two addition arguments
        """
        stride = self.kwargs['stride']
        sliding_window_size = self.kwargs['sliding_window_size']
        grad = self.args[1].data * grad_output
        result = np.zeros_like(self.args[0].data)
        shape = result.shape
        for i in range(shape[0]):
            for i1 in range(shape[1]):
                for i2 in range(shape[2]):
                    for i3 in range(shape[3]):
                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = i1 * stride
                        vert_end = vert_start + sliding_window_size
                        horiz_start = i2 * stride
                        horiz_end = horiz_start + sliding_window_size
                        a_prev_slice = grad[i, vert_start:vert_end, horiz_start:horiz_end, i3]
                        result[i, i1, i2, i3] = np.max(a_prev_slice)

        return [result]
