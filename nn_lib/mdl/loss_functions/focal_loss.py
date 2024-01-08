from nn_lib.mdl.loss_functions.loss import Loss
from nn_lib import Tensor
import nn_lib.tensor_fns as F


class FocalLoss(Loss):
    """
    Focal loss
    Similar to this https://arxiv.org/pdf/1708.02002.pdf
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
        eps = Tensor(1e-8, requires_grad=False)
        one = Tensor(1, requires_grad=False)

        y = F.sigmoid(prediction_logits) + eps
        # :TODO implement F.pow(x,y) and (1 - y) ** gamma
        loss = - ((one - y) * (one - y) * target * F.log(y) + (one - target) * F.log(one - y))
        return loss


'''
def focal_loss(y_pred, y_real, eps = 1e-8, gamma = 2):
    y =  y_pred.sigmoid() + eps
    loss = -((1 - y) ** gamma * y_real * y.log() + (1 - y_real) * (1 - y).log())
    return loss.mean()

'''
