from typing import Tuple
import numpy as np


class Dataset:
    """
    A very simple dataset class
    """
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve a pair of (data, label) by a given index
        :param index: index of the dataset element to retrieve
        :return: a tuple containing a data sample and  its corresponding label
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """
        :return: number of available samples in the dataset
        """
        raise NotImplementedError
