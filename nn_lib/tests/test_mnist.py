import unittest
import numpy as np
import torch

from nn_lib import Tensor
import nn_lib.tensor_fns as F

from mdl import CELoss


class TestSoftMax(unittest.TestCase):
    """
    Tests responsible for correctness of softmax of nn_lib.math_fns.Functions
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


class TestCELoss(unittest.TestCase):
    """
    Tests responsible for correctness of mdl.ce_loss
    """

    def test_CE_loss_1(self):
        loss = CELoss()
        prediction_logits, targets = Tensor(np.array([0, 0, 0, 1])), Tensor(np.array([0, 0, 0, 1]))
        loss_value = loss.forward(prediction_logits, targets)
        np.testing.assert_almost_equal(loss_value.data, 0., 1)

    
    def test_CE_loss_4(self):
        loss = CELoss()
        
        prediction_logits, targets = Tensor(np.array([0.228, 0.619, 0.153])), Tensor(np.array([0, 1, 0]))
        sm_predict = F.softmax(prediction_logits)
        loss_value = loss(prediction_logits, targets)
        np.testing.assert_almost_equal(loss_value.data, 0.479, 3)
                                                              

    def test_CE_loss_3(self):
        loss = CELoss()
        prediction_logits, targets = Tensor([0.1582, 0.4139, 0.2287]), Tensor(np.array([0.0, 1.0, 0.0]))
        sm_predict = F.softmax(prediction_logits)
        loss_value = loss(sm_predict, targets)
        np.testing.assert_almost_equal(loss_value.data, 0.43, 2)
                                 


    def test_CE_loss_2(self):
        data = np.array([0, 0, 0, 1])
        predict = Tensor(data, requires_grad=False)
        self.assertEqual(t.data.shape , (4,))
        ret = F.softmax(predict)
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
