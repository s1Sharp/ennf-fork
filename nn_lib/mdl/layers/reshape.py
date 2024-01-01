from typing import Tuple
import numpy as np

from nn_lib.mdl.module import Module
from nn_lib import Tensor
import nn_lib.tensor_fns as F


class Reshape(Module):
    """

    """
    def __init__(self, in_dim: int, out_dim: int, activation_fn: str = 'relu'):
        raise NotImplementedError


    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


    def __str__(self):
        result = f'Reshape layer'
        return result
