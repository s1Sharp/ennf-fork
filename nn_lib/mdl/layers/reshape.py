from typing import Tuple

from nn_lib.mdl.module import Module
from nn_lib import Tensor
import nn_lib.tensor_fns as F


class Reshape(Module):
    """

    """
    def __init__(self, shape:Tuple[int]):
        self.shape = shape


    def forward(self, x: Tensor) -> Tensor:
        result = F.reshape(x,shape=self.shape)
        return result


    def __str__(self):
        result = f'Reshape layer to shape {self.shape}'
        return result
