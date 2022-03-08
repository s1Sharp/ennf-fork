import numpy as np


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
