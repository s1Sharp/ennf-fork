from typing import Tuple
import numpy as np

from nn_lib.mdl.module import Module
from nn_lib import Tensor
import nn_lib.tensor_fns as F


class Linear(Module):
    """
    Linear module is a building block of multi-layer perceptron neural network that performs a linear transform of the
    data batch
    """
    def __init__(self, in_dim: int, out_dim: int, activation_fn: str = 'relu'):
        """
        Create a linear module similar to https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        :param in_dim: number of input dimensions of the layer
        :param out_dim: number of output dimensions of the layer
        :param activation_fn: activation function to apply after linear transformation, either 'relu' or 'none'
        """
        assert activation_fn in ('relu', 'none')
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation_fn = activation_fn

        scale = np.sqrt(1 / self.in_dim)
        self.weight = Tensor(self.init_parameter((self.in_dim, self.out_dim), scale), requires_grad=True)
        self.bias = Tensor(self.init_parameter((1, self.out_dim), scale), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply a linear transform to the input
        :param x: an input of the shape (B, self.in_dim), where B is the batch size
        :return: an output of the layer of the shape (B, self.out_dim), where B is the batch size
        """
        # weight already .T
        result = F.mat_mul(x , self.weight) + self.bias
        if self.activation_fn == 'relu':
            result = F.relu(result)
        return result

    @staticmethod
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
        result = f'Linear layer: size ({self.in_dim}, {self.out_dim}), activation {self.activation_fn}'
        return result
