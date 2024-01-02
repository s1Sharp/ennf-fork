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
    def __init__(self, in_dim: int, out_dim: int):
        raise NotImplementedError


    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


    def __str__(self):
        result = f'Max pooling 2d layer'
        return result
