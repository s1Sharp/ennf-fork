from typing import Tuple
import numpy as np

from nn_lib.mdl.module import Module
from nn_lib import Tensor
import nn_lib.tensor_fns as F


class BatchNorm(Module):
    """
    https://d2l.ai/chapter_convolutional-modern/batch-norm.html
    Linear module is a building block of multi-layer perceptron neural network that performs a linear transform of the
    data batch
    """
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None
    ) -> None:
        """
        Create a batch norm module similar to https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
        :param in_dim: number of input dimensions of the layer
        :param out_dim: number of output dimensions of the layer
        :param activation_fn: activation function to apply after linear transformation, either 'relu' or 'none'
        """
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply a linear transform to the input
        :param x: an input of the shape (B, self.in_dim), where B is the batch size
        :return: an output of the layer of the shape (B, self.out_dim), where B is the batch size
        """
        # weight already .T
        raise NotImplementedError

    def __str__(self):
        result = f'Batch size layer: size ({self.in_dim}, {self.out_dim}), activation {self.activation_fn}'
        return result
