from typing import Tuple, Union
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

from nn_lib.data import Dataset


DEFAULT_DATASET_PARAMETERS = dict(
    blobs=dict(centers=np.array([[0, 0], [3, 5]]), cluster_std=np.array([1, 2])),
    moons=dict(noise=0.1),
    circles=dict(noise=0.1, factor=0.55)
)


class ToyDataset(Dataset):
    """
    A simple binary classification dataset consisting of two 2D point clusters, each cluster representing a
    classification category; points are artificially generated
    Clusters can be arranged in three different configurations:
        - blobs -- clusters represent two different Gaussian distributions
        - moons -- clusters represent points from two half circles
        - circles -- clusters represent two circles one inside another
    """
    def __init__(self, n_samples: int, structure: str = 'blobs', seed: int = 0, **kwargs):
        """
        Create a dataset
        :param n_samples: total number of points in the dataset split equally among two categories
        :param structure: arrangement of the points either 'blobs', 'moons' or 'circles'
        :param seed: random generator seed
        :param kwargs: optional keyword arguments for the dataset generation
        """
        assert structure in ('blobs', 'moons', 'circles')

        dataset_parameters = {}
        for k, v in DEFAULT_DATASET_PARAMETERS[structure].items():
            dataset_parameters[k] = kwargs[k] if k in kwargs else v

        self.n_samples = n_samples
        self.structure = structure
        if self.structure == 'blobs':
            self.data, self.labels = datasets.make_blobs(
                n_samples=n_samples, n_features=2, random_state=seed, **dataset_parameters)
        elif self.structure == 'moons':
            self.data, self.labels = datasets.make_moons(
                n_samples=n_samples, random_state=seed, **dataset_parameters)
        else:
            self.data, self.labels = datasets.make_circles(
                n_samples=n_samples, random_state=seed, **dataset_parameters)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        result = self.data[index], self.labels[index]
        return result

    def __len__(self) -> int:
        return self.n_samples

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
        positive_mask = self.labels == 1
        if predictions is None:
            plt.scatter(self.data[positive_mask][:, 0], self.data[positive_mask][:, 1], color='green',
                        label='positive')
            plt.scatter(self.data[~positive_mask][:, 0], self.data[~positive_mask][:, 1], color='blue',
                        label='negative')
        else:
            pred_positive_mask = predictions == 1
            tp_mask = positive_mask & pred_positive_mask
            tn_mask = (~positive_mask) & (~pred_positive_mask)
            fp_mask = (~positive_mask) & pred_positive_mask
            fn_mask = positive_mask & (~pred_positive_mask)
            plt.scatter(self.data[tp_mask][:, 0], self.data[tp_mask][:, 1], color='green',
                        label='true positive')
            plt.scatter(self.data[tn_mask][:, 0], self.data[tn_mask][:, 1], color='blue',
                        label='true negative')
            plt.scatter(self.data[fp_mask][:, 0], self.data[fp_mask][:, 1], color='orange',
                        label='false positive')
            plt.scatter(self.data[fn_mask][:, 0], self.data[fn_mask][:, 1], color='magenta',
                        label='false negative')

        plt.legend()
        plt.tight_layout()
        plt.gca().set_aspect('equal')
        plt.grid()
        plt.show()
        plt.cla()


if __name__ == '__main__':
    dataset = ToyDataset(1000, 'blobs')
    #dataset = ToyDataset(1000, 'moons')
    #dataset = ToyDataset(1000, 'circles')
    dataset.visualize()
