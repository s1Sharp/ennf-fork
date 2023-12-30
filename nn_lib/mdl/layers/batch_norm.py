from typing import Tuple
import numpy as np

from nn_lib.mdl.module import Module
from nn_lib import Tensor, SetGrad
import nn_lib.tensor_fns as F


class BatchNorm(Module):
    """
    abstract batch normalization layer
    """

    def __init__(
            self,
            num_dims: int,
            out_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            track_running_stats: bool = True
    ) -> None:
        """
        Create a batch norm module similar to https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
        :param num_features: number of input dimensions of the layer
        # num_features: the number of outputs for a fully connected layer or the
        # number of output channels for a convolutional layer. num_dims: 2 for a
        # fully connected layer and 4 for a convolutional layer
        """
        if num_dims == 2:
            shape = (1, out_features)
        else:
            shape = (1, out_features, 1, 1)
            # The scale parameter and the shift parameter (model parameters) are
            # initialized to 1 and 0, respectively
        self.gamma = Tensor(np.ones(shape), requires_grad=False)
        self.beta = Tensor(np.zeros(shape), requires_grad=False)
        # The variables that are not model parameters are initialized to 0 and
        # 1
        self.moving_mean = Tensor(np.zeros(shape), requires_grad=False)
        self.moving_var = Tensor(np.ones(shape), requires_grad=False)
        self.eps = Tensor(np.full(shape=shape,fill_value=eps), requires_grad=False)
        self.momentum = Tensor(np.full(shape=shape,fill_value=momentum), requires_grad=False)
        self.out_features = out_features
        self.num_dims = num_dims



    def forward(self, x: Tensor) -> Tensor:
        """
        Apply a linear transform to the input
        :param x: an input of the shape (B, self.in_dim), where B is the batch size
        :return: an output of the layer of the shape (B, self.out_dim), where B is the batch size
        """
        Y, self.moving_mean, self.moving_var = self._batch_norm(
            x, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=self.eps, momentum=self.momentum)
        return Y

    def __str__(self):
        result = f'Batch size layer: size ({self.out_features}, {self.num_dims})'
        return result

    def _batch_norm(self,X: Tensor, gamma: Tensor, beta: Tensor, moving_mean: Tensor,
                    moving_var: Tensor, eps: Tensor, momentum: Tensor):
        # Use is_grad_enabled to determine whether we are in training mode
        if not SetGrad.is_grad_enabled():
            # In prediction mode, use mean and variance obtained by moving average
            X_hat = (X - moving_mean) / F.sqrt(moving_var + eps)
        else:
            assert len(X.shape) in (2, 4)
            if len(X.shape) == 2:
                # When using a fully connected layer, calculate the mean and
                # variance on the feature dimension
                mean = F.reduce(X, axis=0, reduction='mean')
                var = F.reduce(F.pow2(X - mean), axis=0, reduction='mean')
            else:
                # When using a two-dimensional convolutional layer, calculate the
                # mean and variance on the channel dimension (axis=1). Here we
                # need to maintain the shape of X, so that the broadcasting
                # operation can be carried out later
                mean = F.reduce(X, axis=(0, 2, 3), reduction='mean', keepdims=True)
                var = F.reduce(F.pow2(X - mean), axis=(0, 2, 3), reduction='mean', keepdims=True)
            # In training mode, the current mean and variance are used
            X_hat = (X - mean) / F.sqrt(var + eps)
            # Update the mean and variance using moving average
            one = Tensor(np.ones(momentum.shape), requires_grad=False)
            moving_mean = (one - momentum) * moving_mean + momentum * mean
            moving_var = (one - momentum) * moving_var + momentum * var

        result = gamma * X_hat + beta  # Scale and shift
        return result, moving_mean, moving_var

