from typing import Union, Tuple, List
import numpy as np

from nn_lib import Tensor
from nn_lib.mdl import Module


class MnistMLPClassifier(Module):
    """
    Class representing a multilayer perceptron network for solving binary classification task
    """

    def __init__(self, layers: Union[Tuple[Module, ...], List[Module]]):
        """
        Creates binary MLP classifier
        :param in_features: number of feature in the input data
        :param hidden_layer_sizes: number of neurons in hidden layers of MLP
        """

        self._parameters = []
        self.layers = layers
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
            for p in layer.parameters():
                self._parameters.append(p)
        return

    def forward(self, x: Tensor) -> Tensor:
        """
        Pass an input through the network layers obtaining the prediction logits; later still need to apply
        sigmoid function to obtain confidence values from [0, 1]
        :param x: input data batch of the shape (B, self.in_features)
        :return: prediction batch of logits of the shape (B,)
        """
        predictions = x
        for layer in self.layers:
            predictions = layer.forward(predictions)
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