"""
    https://d2l.ai/chapter_convolutional-modern/batch-norm.html
"""

from typing import Tuple
import numpy as np

from nn_lib.mdl.module import Module
from nn_lib import Tensor, SetGrad
import nn_lib.tensor_fns as F

"""
    https://d2l.ai/chapter_convolutional-modern/batch-norm.html
"""

from typing import Tuple
import numpy as np

from nn_lib.mdl.layers.batch_norm import BatchNorm
from nn_lib import Tensor
import nn_lib.tensor_fns as F


class BatchNorm2d(BatchNorm):
    def __init__(
            self,
            out_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            track_running_stats: bool = True
    ) -> None:
        """
        Create a batch norm module similar to https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
        :param in_dim: number of input dimensions of the layer
        :param out_dim: number of output dimensions of the layer
        :param activation_fn: activation function to apply after linear transformation, either 'relu' or 'none'
        Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
        """
        super().__init__(num_dims=4,out_features=out_features,eps=eps,momentum=momentum)

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x=x)

    def _check_input_dim(self, x: Tensor):
        if len(x.shape) != 2 and len(x.shape) != 3:
            raise ValueError(
                f"expected 2D or 3D input (got {len(x.shape)}D input)"
            )

    def __str__(self):
        result = f'Batch norm 1d layer'
        return result


    def _batch_norm(self,X: Tensor, gamma: Tensor, beta: Tensor, moving_mean: Tensor,
                    moving_var: Tensor, eps: Tensor, momentum: Tensor):
        # Use is_grad_enabled to determine whether we are in training mode
        if SetGrad.is_grad_enabled():
            assert len(X.shape) == 4
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
        else:
            # In prediction mode, use mean and variance obtained by moving average
            X_hat = (X - moving_mean) / F.sqrt(moving_var + eps)

        result = gamma * X_hat + beta  # Scale and shift
        return result, moving_mean, moving_var

    def __str__(self):
        result = f'Batch norm 2d'
        return result
