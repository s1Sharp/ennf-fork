from nn_lib.mdl.loss import Loss
from nn_lib import Tensor
import nn_lib.tensor_fns as F


class BCELoss(Loss):
    """
    Binary cross entropy loss
    Similar to this https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
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
        # l = - (y * log(x) + (1-y) * log(1-x)) 
        x = prediction_logits
        y = target
        ones_Tensor = Tensor(1, requires_grad=False)
        max_log_Tensor = Tensor(self.MAX_LOG, requires_grad=False)
        x = F.clip(x, -max_log_Tensor , max_log_Tensor)

        log1 = -F.log(ones_Tensor+F.exp(-x))
        log2 = F.log(F.exp(-x)) - F.log(ones_Tensor+F.exp(-x))

        result = -(y * log1 + (ones_Tensor - y) * log2)

        if self.reduce:
            return F.reduce(result)
        return result