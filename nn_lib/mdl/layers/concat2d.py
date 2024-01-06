from nn_lib.mdl.module import Module
from nn_lib import Tensor
import nn_lib.tensor_fns as F


class Concat2d(Module):
    """
    Concatinate 2 layers
    """
    def __init__(self, axis:int=-1):
        self.axis = axis


    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        result = F.concat2d(x, y, axis=self.axis)
        return result


    def __str__(self):
        result = f'Concatination of 2 2-d layers in axis {self.axis}'
        return result
