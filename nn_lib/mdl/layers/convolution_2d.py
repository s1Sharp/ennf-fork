from typing import Tuple
import numpy as np

from nn_lib.mdl.module import Module
from nn_lib import Tensor
import nn_lib.tensor_fns as F


class Conv2d(Module):
    """
    2d convolution layer
    Basically convolve 2-dim input
    """
    def __init__(self, in_dim: int, out_dim: int, stride:int = 1, padding: int = 0):
        self.in_dim = in_dim
        self.out_dim = out_dim
        scale = np.sqrt(1 / self.in_dim)
        # :TODO shapes are WRONG
        self.b = Tensor(self.init_parameter((self.in_dim, self.out_dim), scale), requires_grad=True)   # :TODO figure out dims
        self.W = Tensor(self.init_parameter((self.in_dim, self.out_dim), scale), requires_grad=True)   # :TODO figure out dims
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        # :TODO multiple conv cores
        result = F.conv2d(x,self.W,stride=self.stride,padding=self.padding) + self.b
        return result

    def init_parameter(shape: Tuple[int, int], scale: float) -> np.ndarray:
        """
        Used for initializing weight or bias parameters
        :param shape: shape of the parameter
        :param scale: scale of the parameter
        :return: initialized parameter
        """
        result = np.random.uniform(low=-scale, high=scale, size=shape)
        return result

    def __str__(self):
        result = f'Convilution 2d layer'
        return result
