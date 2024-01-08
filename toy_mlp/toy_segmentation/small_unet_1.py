from typing import Union, Tuple, List
import numpy as np

from nn_lib import Tensor
from nn_lib.mdl import Module

from nn_lib.mdl.layers import *
import nn_lib.tensor_fns as F

class SmallUNet1(Module):
    """
    Class representing a multilayer perceptron network for solving binary classification task
    Number of filters K
    their spatial extent K(kernel)
    the stride S
    the amount of zero padding P
    W2=(W1−K+2P)/S+1
    """

    def __init__(self,kernum:int=32):
        """
        Creates binary MLP classifier
        :param in_features: number of feature in the input data
        :param hidden_layer_sizes: number of neurons in hidden layers of MLP
        """
        self._parameters = []
        self.enc_conv0 = [
            Conv2d(3, kernum, 3, 1, 1),
            Relu(),
            Conv2d(kernum, kernum, 3, 1, 1),
            Relu(),
        ]
        self.pool0 = MaxPool2d(2, 2) # 32 -> 16

        self.bottle_neck = [
            Conv2d(kernum, kernum, 3, 1, 1),
            Relu(),
            Conv2d(kernum, kernum, 3, 1, 1),
            Relu()
        ]
        self.unpool0 = MaxUnpool2d(2, 2)  # 8 -> 16
        self.concat0 = Concat2d()
        self.dec_conv0 = [
            Conv2d(kernum*2, kernum, 3, 1, 1),
            Relu(),
            Conv2d(kernum, kernum, 3, 1, 1),
            Relu()
        ]

        self.last = Conv2d(kernum, 1, 3, 1, 1)

        self.layers = [
            self.enc_conv0, self.bottle_neck ,self.dec_conv0, [self.last]
        ]
        self._fill_parameters()

    def parameters(self) -> List[Tensor]:
        result = self._parameters.copy()
        return result

    def _fill_parameters(self) -> None:
        """
        Adds a single layer to the network and updates internal list of parameters accordingly (both weight and bias)
        :param in_dim: number of features returned by the previous layer
        :param out_dim: number of features for the added layer to return
        :param activation_fn: activation function to apply to the outputs of the added layer
        :return: None
        """
        for layer in self.layers:
            for l in layer:
                for p in l.parameters():
                    self._parameters.append(p)
        return

    def _forward(self ,x: Tensor,block: List[Module]) -> Tensor:
        prediction = x
        for module in block:
            prediction = module.forward(prediction)
        return prediction

    def forward(self, x: Tensor) -> Tensor:
        """
        Pass an input through the network layers obtaining the prediction logits; later still need to apply

        """
        pre_e0 = self._forward(x,self.enc_conv0)
        e0 = self.pool0.forward(pre_e0)
        ind0 = self.pool0.get_mask()

        b = self._forward(e0, self.bottle_neck)

        d0 = self.unpool0.forward(b,ind0)
        cat0 = self.concat0.forward(pre_e0, d0)
        pre_d0 = self._forward(cat0, self.dec_conv0)

        predictions = self.last.forward(pre_d0)

        return predictions

    def parameter_count(self) -> int:
        """
        Count total number of trainable parameters of the network
        :return: number of trainable parameters of the network
        """
        result = 0
        for param in self.parameters():
            result += np.prod(param.shape)
        return result

    def __str__(self) -> str:
        result = '\n'.join(map(str, self.layers)) + f'\nTotal number of parameters: {self.parameter_count()}'
        return result
