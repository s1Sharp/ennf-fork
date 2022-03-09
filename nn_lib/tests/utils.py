from typing import Union, Iterable
import numpy as np


def assert_almost_equal(a: Union[np.ndarray, Iterable, float, int], b: Union[np.ndarray, Iterable, float, int]) -> None:
    """
    Assert whether two values/arrays are equal up to 4 decimal places
    :param a: the first value to compare
    :param b: the second value to compare
    :return: None
    """
    a_ = np.array(a, np.float32)
    b_ = np.array(b, np.float32)
    np.testing.assert_almost_equal(a_, b_, 4)


class seeded_random:
    """
    Creates a context manager with fixed numpy random state defined by seed
    """
    def __init__(self, seed: int):
        """
        :param seed: seed to use within the context; if None, random state is the same as outside the context
        """
        self.seed = seed

    def __enter__(self):
        if self.seed is None:
            return
        self.after_seed = np.random.randint(1 << 30)
        np.random.seed(self.seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.seed is None:
            return
        np.random.seed(self.after_seed)
