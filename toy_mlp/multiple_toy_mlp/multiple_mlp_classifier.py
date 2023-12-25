from typing import Union, Tuple, List
import numpy as np

from nn_lib import Tensor
from nn_lib.mdl import Module
from nn_lib.mdl.layers import Linear


class MultipleMLPClassifier(Module):
    """
    Class representing a multilayer perceptron network for solving binary classification task
    """
    def __init__(self, in_features: int, hidden_layer_sizes: Union[Tuple[int, ...], List[int]]):
        """
        Creates binary MLP classifier
        :param in_features: number of feature in the input data
        :param hidden_layer_sizes: number of neurons in hidden layers of MLP
        """
        self.in_features = in_features
        self.hidden_layer_sizes = hidden_layer_sizes

        self._parameters = []
        self.layers = []        # type: List[Linear]
        self._build_layers()

    def parameters(self) -> List[Tensor]:
        result = self._parameters.copy()
        return result

    def _build_layers(self) -> None:
        """
        Create hidden layers of the MLP
        First linear layer transforms input data into self.hidden_layer_sizes[0] dimensions
        Last linear layer transforms features from self.hidden_layer_sizes[-1] dimensions into a single dimension with
        no activation function
        Output of the last layer will later be used as an argument to the sigmoid function
        :return: None
        """
        self._add_layer(self.in_features, self.hidden_layer_sizes[0], 'relu')

        for in_dim_index, in_dim in enumerate(self.hidden_layer_sizes[0:-2]):
            out_dim = self.hidden_layer_sizes[in_dim_index+1]
            self._add_layer(in_dim, out_dim, 'relu')

        self._add_layer(self.hidden_layer_sizes[-2], self.hidden_layer_sizes[-1],'none')
        return

    def _add_layer(self, in_dim: int, out_dim: int, activation_fn: str) -> None:
        """
        Adds a single layer to the network and updates internal list of parameters accordingly (both weight and bias)
        :param in_dim: number of features returned by the previous layer
        :param out_dim: number of features for the added layer to return
        :param activation_fn: activation function to apply to the outputs of the added layer
        :return: None
        """
        self.layers.append(Linear(in_dim, out_dim, activation_fn))
        self._parameters.append(self.layers[-1].weight)
        self._parameters.append(self.layers[-1].bias)
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
