from nn_lib.mdl.loss_functions.loss import Loss
from nn_lib import Tensor
import nn_lib.tensor_fns as F


class CELoss(Loss):
    """
    Cross entropy loss
    Similar to this https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    """

    # In order to avoid over- or underflow we clip prediction logits into [-MAX_LOG, MAX_LOG]
    MAX_LOG = 50

    def forward(self, prediction_logits: Tensor, target: Tensor) -> Tensor:
        """
        Compute a loss Tensor based on logit predictions and ground truth labels

        :param prediction_logits: prediction logits returned by a model (i.e. sigmoid argument) of shape (B,)
        :param target: binary ground truth labels of shape (B,)
        :return: a loss Tensor; if reduction is True, returns a scalar, otherwise a Tensor of shape (B,) -- loss value
            per batch element
        """
        x = prediction_logits
        y = target

        x = F.clip(x, Tensor(-self.MAX_LOG, True), Tensor(self.MAX_LOG, True))
        ax = 1
        result = F.log(F.reduce(F.exp(x), axis=ax, reduction='sum', keepdims=True)) - \
                 F.reduce(x * y, axis=ax,reduction='sum', keepdims=True)

        if self.reduce:
            return F.reduce(result)
        return result
