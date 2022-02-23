from nn_lib.mdl.module import Module
from nn_lib import Tensor


class Loss(Module):
    """
    A base loss module
    """
    def __init__(self, reduce: bool = True):
        """
        Create a loss module
        :param reduce: whether to reduce per-batch element losses by averaging them
        """
        self.reduce = reduce

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """
        Compute a loss value from predictions and targets
        :param prediction: result predicted by a model
        :param target: ground truth label for the predicted data sample
        :return:
        """
        raise NotImplementedError
