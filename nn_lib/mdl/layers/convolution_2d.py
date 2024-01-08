from typing import Tuple, List
import numpy as np

from nn_lib.mdl.module import Module
from nn_lib import Tensor
import nn_lib.tensor_fns as F


class Conv2d(Module):
    """
    2d convolution layer
    Basically convolve 2-dim input
    """
    def __init__(self, in_dim: int, out_dim: int, kernel_size:int, stride:int = 1, padding: int = 0):
        self.in_dim = in_dim
        self.out_dim = out_dim
        scale = np.sqrt(1 / self.in_dim)
        self.bias = Tensor(self.init_parameter(shape=(self.out_dim), scale=scale), requires_grad=True)   # :TODO figure out dims
        self.weight = Tensor(self.init_parameter((kernel_size, kernel_size, self.in_dim, self.out_dim), scale), requires_grad=True)   # :TODO figure out dims
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        conv = F.conv2d(x,self.weight ,stride=self.stride,padding=self.padding)
        result = conv + self.bias
        return result

    def parameters(self) -> List[Tensor]:
        return [self.weight, self.bias]

    def init_parameter(self, shape, scale: float) -> np.ndarray:
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
