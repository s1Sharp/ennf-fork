from nn_lib.optim.optimizer import Optimizer

class Scheduler:
    """

    """

    def __init__(self, optimizer:Optimizer, last_epoch:int=-1, verbose:bool=False):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.verbose = verbose
        self.state_dict = {}
        self.last_lr = 0
        self.cur_step = 0

    def load_state_dict(self) -> dict:
        return self.state_dict

    def step(self):
        if self._need_to_update_lr():
            self._update_lr()
        self._make_step()

    def _make_step(self):
        self.cur_step += 1

    def get_last_lr(self) -> float:
        raise NotImplementedError

    def _update_lr(self):
        raise NotImplementedError


    def _need_to_update_lr(self) -> bool:
        raise NotImplementedError
