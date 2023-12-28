from nn_lib.mdl.module import Module
from nn_lib import Tensor
import nn_lib.tensor_fns as F


class Sigmoid(Module):
    def __init__(self):
        """
        Create a sigmoid module similar to https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        """

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply a sigmoid function to the input
        :param x: an input of the shape (B, self.in_dim), where B is the batch size
        :return: an output of the sigmoid function of the shape (B, self.out_dim), where B is the batch size
        """
        # weight already .T
        result = F.sigmoid(x)
        return result


    def __str__(self):
        result = f'Sigmoid function layer'
        return result
