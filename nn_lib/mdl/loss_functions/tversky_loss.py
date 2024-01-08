from nn_lib.mdl.loss_functions.loss import Loss
from nn_lib import Tensor
import nn_lib.tensor_fns as F


class TverskyLoss(Loss):
    """
    Focal loss
    Similar to this https://arxiv.org/abs/1706.05721
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
        alpha = Tensor(0.3, requires_grad=False)
        beta = Tensor(0.7, requires_grad=False)

        TP = F.reduce(prediction_logits * target, reduction='sum')  # True Positive
        FP = F.reduce((one - target) * prediction_logits, reduction='sum')  # False Positive
        FN = F.reduce(target * (one - prediction_logits), reduction='sum')  # False Negative
        loss = one - (TP + eps) / (TP + alpha * FP + beta * FN + eps)
        return loss


'''
def tversky_loss(y_pred, y_real, smooth=1e-8, alpha=0.3, beta=0.7):
    y_pred = y_pred.sigmoid().view(-1)
    y_real = y_real.view(-1)

    TP = (y_pred * y_real).sum() #True Positive
    FP = ((1-y_real) * y_pred).sum() #False Positive
    FN = (y_real * (1-y_pred)).sum() #False Negative

    return 1 - (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
'''
