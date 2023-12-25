from typing import Tuple, Union
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

from nn_lib.data import Dataset

class ToyDatasetMultiple(Dataset):
    """
    A simple multiple classification dataset consisting of two 2D point clusters, each cluster representing a
    classification category; points are artificially generated
    Clusters can be arranged in three different configurations:
        - blobs -- clusters represent two different Gaussian distributions
        - moons -- clusters represent points from two half circles
        - circles -- clusters represent two circles one inside another
    """
    def __init__(self, n_samples: int, structure: str = 'blobs', seed: int = 0, classes: int = 2):
        """
        Create a dataset
        :param n_samples: total number of points in the dataset split equally among two categories
        :param structure: arrangement of the points either 'blobs', 'moons' or 'circles'
        :param seed: random generator seed
        :param kwargs: optional keyword arguments for the dataset generation
        """
        assert structure in ('blobs', 'moons', 'circles')
        self.FACTOR = 0.55
        self.NOISE = 0.1
        self.classes = classes // 2 * 2
        self.n_samples = n_samples
        self.structure = structure
        self.plainLabels=None

        if self.structure == 'blobs':
            self.data, self.labels = datasets.make_blobs(
                n_samples=n_samples, n_features=2, random_state=seed,centers=self.classes, cluster_std = 0.7)
        elif self.structure == 'moons':
            self.data, self.labels = datasets.make_moons(
                n_samples=n_samples//(self.classes)*2, random_state=seed, noise=self.NOISE)
            for i in range(int(self.classes//2-1) ):
                data, labels = datasets.make_moons(
                    n_samples=n_samples//(self.classes)*2, random_state=i+1, noise=self.NOISE)
                self.data = np.concatenate([self.data,data*(2**(i+1))+(i+3)],axis=0)
                self.labels = np.concatenate([self.labels, labels+((i+1)*2)], axis=0)
        else:
            self.data, self.labels = datasets.make_circles(
                n_samples=n_samples//(self.classes)*2, random_state=seed,factor=self.FACTOR, noise=self.NOISE)
            for i in range(int(self.classes // 2 - 1)):
                data, labels = datasets.make_circles(
                n_samples=n_samples//(self.classes)*2, random_state=i + 1, noise=0.05)
                self.data = np.concatenate([self.data, data * (2 ** (i + 3))], axis=0)
                self.labels = np.concatenate([self.labels, labels + ((i + 1) * 2)], axis=0)

        self.label_to_one_hot_ecoding()


    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        result = self.data[index], self.labels[index]
        return result

    def __len__(self) -> int:
        return self.n_samples

    def label_to_one_hot_ecoding(self):
        self.plainLabels = self.labels
        ohe = OneHotEncoder()
        self.labels = ohe.fit_transform(self.labels.reshape(-1,1)).toarray()

    def label_from_one_hot_encoding(self):
        pass

    def visualize(self, predictions: Union[np.ndarray, None] = None) -> None:
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
            for i in range(self.classes):
                positive_mask = self.plainLabels == i
                plt.scatter(self.data[positive_mask][:, 0], self.data[positive_mask][:, 1],
                            label='class' + str(i))
            plt.legend(loc='best')

        else:
            for i in range(self.classes):
                lbl = np.zeros(shape=(self.classes),dtype=np.float64)
                lbl[i] = 1.0
                positive_mask = np.logical_and.reduce(self.labels == lbl, axis=1)
                pred_positive_mask = np.array(predictions) == i
                tp_mask = positive_mask & pred_positive_mask

                fp_mask = (~positive_mask) & pred_positive_mask

                plt.scatter(self.data[tp_mask][:, 0], self.data[tp_mask][:, 1],
                            label='tp'+str(i))
                plt.scatter(self.data[fp_mask][:, 0], self.data[fp_mask][:, 1],
                            label='fp'+str(i))

        plt.legend(loc='best')
        plt.tight_layout()
        plt.gca().set_aspect('equal')
        plt.grid()
        plt.show()
        plt.cla()


if __name__ == '__main__':
    dataset = ToyDatasetMultiple(1000, 'blobs',classes=4)
    #dataset = ToyDatasetMultiple(10000, 'moons',classes=9)
    #dataset = ToyDatasetMultiple(10000, 'circles',classes=4)
    dataset.visualize()
