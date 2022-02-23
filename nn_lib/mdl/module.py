from typing import Union, Tuple

from nn_lib import Tensor


class Module:
    """
    Module should implement some stateful function accepting ang returning one or multiple Tensors
    """
    def forward(self, *args: Tensor, **kwargs) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        The method performing the main computing logic of a module
        :param args: any number of Tensor arguments
        :param kwargs: some additional non-tensor keyword arguments
        :return: resulting Tensor(s)
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """
        A module can additionally be called as a callable object
        """
        return self.forward(*args, **kwargs)
