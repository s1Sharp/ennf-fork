from nn_lib.mdl.loss_functions.loss import Loss
from nn_lib import Tensor
import nn_lib.tensor_fns as F


class DiceLoss(Loss):
    """
    Cross entropy loss
    Similar to this https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
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
        gamma = Tensor(1, requires_grad=False)
        smooth = Tensor(1e-6, requires_grad=False)
        shape = target.shape
        nominator = Tensor(2, requires_grad=False) * F.reduce(F.mat_mul(prediction_logits, target), reduction='sum',
                                                              axis=1) + smooth
        # :TODO here need to implement pow   denominator = torch.sum(y_pred ** gama) + torch.sum(y_real ** gama) + smooth
        denominator = F.reduce(prediction_logits, reduction='sum', axis=1) + F.reduce(target, reduction='sum',
                                                                                      axis=1) + smooth
        res = Tensor(1, requires_grad=False) - nominator / denominator / Tensor(shape[1], requires_grad=False) / Tensor(
            shape[2], requires_grad=False)
        return res


'''
def dice_loss(y_real, y_pred):
  gama = 1
  smooth = 1e-6
  shape = y_real.squeeze(1).shape
  nominator = 2 * torch.sum(y_pred * y_real) + smooth
  denominator = torch.sum(y_pred ** gama) + torch.sum(y_real ** gama) + smooth
  res = 1 - nominator / denominator / shape[1] / shape[2]
  return res
'''
