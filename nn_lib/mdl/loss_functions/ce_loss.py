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
        tensor_max_log = Tensor(self.MAX_LOG, requires_grad=True)
        sm = F.softmax(prediction_logits)
        l = F.clip(-F.log(sm),-tensor_max_log, tensor_max_log)
        return F.reduce(l*target)
