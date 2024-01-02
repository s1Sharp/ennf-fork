from typing import Tuple, Union
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import pickle, gzip

from nn_lib.data import Dataset


class CIFAR10(Dataset):
    """
    A simple CIFAR10 classification dataset consisting of ten digits

    """
    def __init__(self, ds_type: str = 'test'):
        """
        Init mnist dataset
        """
        # open a file, where you stored the pickled data
        assert ds_type in ('val', 'train')

        self.path = r'cifar-10-batches-py/'
        self.file:str
        self.data: np.ndarray
        self.labels: np.ndarray
        self.n_samples:int = 10000
        self.batch_num = 1

        self.ds_type = ds_type
        # dump information to that file
        if ds_type == 'train':
            self.file = 'data_batch_'

        elif ds_type == 'val':
            self.file = 'test_batch'
            self._load_batch()



    def _load_batch(self):
        full_path = self.path + self.file
        if self.ds_type == 'train':
            full_path += str(self.batch_num)
            self.batch_num += 1

        with open(full_path, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        self.data, self.labels = dict[b'data'], np.array(dict[b'labels'])
        self.n_samples = len(self.labels)
        self.data = self.from1dTo2d()
        self.labels = self.label_to_one_hot_ecoding()

    def label_to_one_hot_ecoding(self):
        self.plainLabels = self.labels
        ohe = OneHotEncoder()
        return ohe.fit_transform(self.labels.reshape(-1, 1)).toarray()

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.ds_type == 'train':
            if index >= self.n_samples * (self.batch_num-1):
                self._load_batch()
        index = index - self.n_samples*(self.batch_num-1)
        result = self.data[index], self.labels[index]
        return result

    def __len__(self) -> int:
        if self.ds_type == 'train':
            return self.n_samples * 5
        return self.n_samples

    def from1dTo2d(self):
        return np.moveaxis(self.data.reshape((self.n_samples,3,32,32)),1,-1)

    def visualize(self, predictions: Union[np.ndarray, None] = None, number: int = -1,
                  show_positive: bool = True) -> None:
        """
        Helper method for visualizing data points with and without predictions
        If predictions are not passed, visualizes points coloring positive and negative categories differently
        If prediction are passed, visualizes points coloring true positive, true negatives, false positives and
        false negative differently
        TODO: enhance visualization by plotting decision boundary and spatial prediction confidence of a classifier
        :param predictions: an array of binary predictions for each data point
        :return: None
        """

        if predictions is None:
            mask = np.unique(self.plainLabels,return_index=True)[1]
        else:
            mask = self.plainLabels == predictions

            if show_positive:
                if 0 <= number <= 9:
                    mask = (self.plainLabels == number) == mask
            else:
                if 0 <= number <= 9:
                    mask = (self.plainLabels == number) == (self.plainLabels != predictions)
                else:
                    mask = self.plainLabels != predictions

        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(self.data[mask][i], interpolation='nearest')
            if predictions is None:
                plt.title("Class {}".format(self.plainLabels[mask][i]))
            if predictions:
                plt.title("Class {}".format(np.array(predictions)[mask][i]))
            plt.axis('off')

        plt.legend()
        plt.tight_layout()
        plt.gca().set_aspect('equal')
        plt.show()
        plt.cla()


if __name__ == '__main__':
    # dataset = ToyDataset(1000, 'blobs')
    # dataset = ToyDataset(1000, 'moons')
    train_dataset = CIFAR10(ds_type='train')
    data, label = next(iter(train_dataset))
    train_dataset.visualize()
