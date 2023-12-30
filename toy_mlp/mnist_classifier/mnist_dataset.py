from typing import Tuple, Union
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import pickle, gzip

from nn_lib.data import Dataset


class MnistDataset(Dataset):
    """
    A simple MNIST classification dataset consisting of ten digits

    """

    def __init__(self, ds_type: str = 'test'):
        """
        Init mnist dataset
        """
        # open a file, where you stored the pickled data
        assert ds_type in ('test', 'val', 'train')

        file = open('mnist.pkl', 'rb')

        # dump information to that file
        data = pickle.load(file, encoding='latin1')
        k = 0
        if ds_type == 'train':
            # train dataset
            k = 0
        elif ds_type == 'test':
            # test dataset
            k = 2
        elif ds_type == 'val':
            k = 1
        self.data, self.labels = data[k]
        self.n_samples = self.labels.shape[0]
        self.label_to_one_hot_ecoding()

    def label_to_one_hot_ecoding(self):
        self.plainLabels = self.labels
        ohe = OneHotEncoder()
        self.labels = ohe.fit_transform(self.labels.reshape(-1, 1)).toarray()

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        result = self.data[index], self.labels[index]
        return result

    def __len__(self) -> int:
        return self.n_samples

    def from1dTo2d(self):
        self.data = self.data.reshape(self.data.shape[0], 28, 28)

    def from2dTo1d(self):
        self.data = self.data.reshape(self.data.shape[0], self.data.shape[1] * self.data.shape[2])

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
            mask = np.arange(12)
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
            plt.imshow(self.data[mask][i].reshape(28, 28), cmap='gray', interpolation='nearest')
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
    train_dataset = MnistDataset(ds_type='train')
    train_dataset.visualize()
