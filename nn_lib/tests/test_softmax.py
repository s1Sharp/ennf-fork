import unittest
import numpy as np
import torch

from nn_lib import Tensor
import nn_lib.tensor_fns as F


class TestSoftMax(unittest.TestCase):
    """
    Tests responsible for correctness of .backward() methods of nn_lib.math_fns.Functions
    """

    def test_softmax_vector(self):
        data = np.array([0, 1.5, 3.2, -5])
        t = Tensor(data, requires_grad=False)
        self.assertEqual(t.data.shape , (4,))
        ret = F.softmax(t)
        np.testing.assert_almost_equal( ret.data,
                                        np.array([3.33100638e-02, 1.49285349e-01, 8.17180146e-01, 2.24441444e-04])
                                        ,
                                        decimal=5)
        np.testing.assert_almost_equal( ret.data.sum(), 1.,
                                        decimal=5)                                        


    def test_softmax_matrix(self):
        data = np.array([[1, 0.5, 0.2, 3], [1,  -1,   7, 3], [2,  12,  13, 3]])
        t = Tensor(data, requires_grad=False)
        self.assertEqual(t.data.shape , (3,4))
        ret = F.softmax(t)
        np.testing.assert_almost_equal( ret.data,
                                        np.array([[  4.48309e-06,   2.71913e-06,   2.01438e-06,   3.31258e-05],
                                                  [  4.48309e-06,   6.06720e-07,   1.80861e-03,   3.31258e-05],
                                                  [  1.21863e-05,   2.68421e-01,   7.29644e-01,   3.31258e-05]])
                                        ,
                                        decimal=5)
        np.testing.assert_almost_equal( ret.data.sum(), 1.,
                                        decimal=5)


if __name__ == '__main__':
    unittest.main()
