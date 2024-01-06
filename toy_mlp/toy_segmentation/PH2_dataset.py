from typing import Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import os
from skimage.transform import resize


from nn_lib.data import Dataset


class PH2(Dataset):
    """
    A simple PH2 classification dataset
    https://www.kaggle.com/datasets/kanametov/ph2dataset
    """

    def __init__(self, ds_type: str = 'test'):
        """
        Init PH2 dataset for segmentation
        """
        # open a file, where you stored the pickled data
        assert ds_type in ('val', 'train')

        images = []
        lesions = []
        self.ds_type = ds_type
        self.test:int = 150
        self.val:int = 50

        self.root = r'C:\Users\Timur\Documents\GitHub\ennf-fork\toy_mlp\toy_segmentation\PH2Dataset'
        for root, dirs, files in os.walk(os.path.join(self.root, 'PH2_Dataset')):
            if root.endswith('_Dermoscopic_Image'):
                images.append(imread(os.path.join(root, files[0])))
            if root.endswith('_lesion'):
                lesions.append(imread(os.path.join(root, files[0])))

        size = (256, 256)
        self.data = [resize(x, size, mode='constant', anti_aliasing=True, ) for x in images]
        self.labels = [resize(y, size, mode='constant', anti_aliasing=False) > 0.5 for y in lesions]
        if self.ds_type == 'test':
            self.data = np.array(self.data[:150])
            self.labels = np.array(self.labels[:150])
        elif self.ds_type == 'val':
            self.data = np.array(self.data[150:])
            self.labels = np.array(self.labels[150:])
        self.n_samples = len(self.data)


    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        index = index - self.n_samples
        result = self.data[index], self.labels[index]
        return result

    def __len__(self) -> int:
        return self.n_samples

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

        #clear_output(wait=True)

        for k in range(6):
            plt.subplot(3, 6, k + 1)
            plt.imshow(self.data[k])
            plt.title('Real')
            plt.axis('off')

            plt.subplot(3, 6, k + 7)
            if predictions:
                plt.imshow(predictions[k], cmap='gray')
            else:
                plt.imshow(self.data[k], cmap='gray')
            plt.title('Output')
            plt.axis('off')

            plt.subplot(3, 6, k + 13)
            plt.imshow(self.labels[k], cmap='gray')
            plt.title('Mask')
            plt.axis('off')

        plt.legend()
        plt.tight_layout()
        plt.gca().set_aspect('equal')
        plt.show()
        plt.cla()


if __name__ == '__main__':
    train_dataset = PH2(ds_type='val')
    train_dataset.visualize()
