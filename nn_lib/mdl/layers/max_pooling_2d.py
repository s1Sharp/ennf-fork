from typing import Tuple
import numpy as np

from nn_lib.mdl.module import Module
from nn_lib import Tensor
import nn_lib.tensor_fns as F


class MaxPool2d(Module):
    """
    Linear module is a building block of multi-layer perceptron neural network that performs a linear transform of the
    data batch
    """

    def __init__(self, sliding_window_size: int = 2, stride: int = 2, padding: int = 0):
        self.sliding_window_size: int = sliding_window_size
        self.padding: int = padding
        self.stride: int = stride
        self._mask: np.ndarray

    def forward(self, x: Tensor) -> Tensor:
        result = F.max_pool_2d(x, sliding_window_size=self.sliding_window_size,
                               stride=self.stride)
        self._mask = result.grad_fn._mask
        return result

    def get_mask(self):
        return self._mask

    def __str__(self):
        result = f'Max pooling 2d layer: \n slWindowSize:{self.sliding_window_size}  stride:{self.stride}'
        return result
