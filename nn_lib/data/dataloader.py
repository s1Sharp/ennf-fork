from typing import Tuple, Iterator
import numpy as np

from nn_lib.data.dataset import Dataset
from nn_lib import Tensor


class Dataloader:
    """
    Dataloader class for iterating over a dataset and combining individual samples into batches
    """
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        """
        Create a dataloader instance
        :param dataset: dataset to iterate through
        :param batch_size: size of batches returned by dataloader
        :param shuffle: whether to shuffle the dataset
        :param drop_last: whether to drop a last batch if its size is less than batch_size
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        dataset_len = len(self.dataset)
        permutation = np.random.permutation(dataset_len) if self.shuffle else np.arange(dataset_len)

        dataset_index = 0
        while dataset_index + (self.batch_size - 1) * int(self.drop_last) < dataset_len:
            data_list = []
            label_list = []
            while dataset_index < dataset_len and len(data_list) < self.batch_size:
                data, label = self.dataset[permutation[dataset_index]]
                data_list.append(data)
                label_list.append(label)
                dataset_index += 1
            data_batch = Tensor(np.stack(data_list))
            label_batch = Tensor(np.stack(label_list))
            yield data_batch, label_batch

    def __len__(self) -> int:
        if self.drop_last:
            result = len(self.dataset) // self.batch_size
        else:
            result = (len(self.dataset) + self.batch_size - 1) // self.batch_size
        return result
