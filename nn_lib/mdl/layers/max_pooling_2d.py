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
    def __init__(self, sliding_window_size: int=2, stride: int=2, padding: int=0):
        self.sliding_window_size=sliding_window_size
        self.padding = padding
        self.stride = stride


    def forward(self, x: Tensor) -> Tensor:
        result = F.max_pool_2d(x,sliding_window_size=self.sliding_window_size,
                             stride=self.stride)
        return result


    def __str__(self):
        result = f'Max pooling 2d layer: \n slWindowSize:{self.sliding_window_size}  stride:{self.stride}'
        return result
