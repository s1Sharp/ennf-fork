from nn_lib.optim.optimizer import Optimizer
from nn_lib.scheduler.scheduler import Scheduler

class MultiStepLR(Scheduler):
    """

    """
    def __init__(self, optimizer:Optimizer, milestones:list=[], gamma:float=0.1, last_epoch:int=-1, verbose:bool=False):
        self.milestones = milestones
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_last_lr(self) -> float:
        return self.last_lr

    def _update_lr(self):
        self.last_lr = self.last_lr * self.gamma
        if self.optimizer.lr:
            self.optimizer.lr = self.last_lr

    def _need_to_update_lr(self) -> bool:
        if self.cur_step in self.milestones:
            return True
        else:
            return False
