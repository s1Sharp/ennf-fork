from typing import Tuple
import numpy as np

from nn_lib.mdl.module import Module
from nn_lib import Tensor
import nn_lib.tensor_fns as F


class Conv2d(Module):
    """

    """
    def __init__(self, in_dim: int, out_dim: int, stride:int = 1, padding: int = 0):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.b = Tensor()
        self.W = Tensor()
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


    def __str__(self):
        result = f'Convilution 2d layer'
        return result
