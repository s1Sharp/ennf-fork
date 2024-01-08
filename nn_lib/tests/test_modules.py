import unittest
import numpy as np

from nn_lib import Tensor
from nn_lib.mdl.layers import Linear
from nn_lib.mdl.loss_functions import BCELoss
from nn_lib.tests.utils import seeded_random


class TestModules(unittest.TestCase):
    """
    Tests covering the implemented nn_lib Modules
    """

    def linear_from_params_test_helper(self, in_dim, out_dim, activation_fn, batch_size):
        linear = Linear(in_dim, out_dim, activation_fn)

        # check that parameters require grad
        self.assertTrue(linear.bias.requires_grad)
        self.assertTrue(linear.weight.requires_grad)

        # check parameter shapes
        self.assertEqual(linear.bias.shape, (1, out_dim))
        self.assertEqual(linear.weight.shape, (in_dim, out_dim))

        # check parameter values
        scale = np.sqrt(1 / in_dim)
        self.assertTrue(linear.bias.data.max() <= scale)
        self.assertTrue(linear.bias.data.max() >= -scale)
        self.assertTrue(linear.weight.data.max() <= scale)
        self.assertTrue(linear.weight.data.max() >= -scale)

        with seeded_random(0):
            x = Tensor(np.random.random((batch_size, in_dim)))
            y = linear(x)

            self.assertTrue(y.requires_grad)
            self.assertEqual(y.shape, (batch_size, out_dim))
            y_true = np.matmul(x.data, linear.weight.data) + linear.bias.data
            if activation_fn == 'relu':
                y_true = np.maximum(y_true, 0)
            else:
                assert activation_fn == 'none'
            np.testing.assert_almost_equal(y.data, y_true, 6)

    def test_linear_1(self):
        in_dim, out_dim = 1, 2
        self.linear_from_params_test_helper(in_dim, out_dim, 'none', 1)

    def test_linear_2(self):
        in_dim, out_dim = 2, 1
        self.linear_from_params_test_helper(in_dim, out_dim, 'none', 1)

    def test_linear_3(self):
        in_dim, out_dim = 2, 2
        self.linear_from_params_test_helper(in_dim, out_dim, 'none', 1)

    def test_linear_4(self):
        in_dim, out_dim = 2, 3
        self.linear_from_params_test_helper(in_dim, out_dim, 'none', 1)

    def test_linear_5(self):
        in_dim, out_dim = 2, 3
        self.linear_from_params_test_helper(in_dim, out_dim, 'none', 4)

    def test_linear_6(self):
        in_dim, out_dim = 10, 20
        self.linear_from_params_test_helper(in_dim, out_dim, 'none', 30)

    def test_linear_relu_1(self):
        in_dim, out_dim = 1, 2
        self.linear_from_params_test_helper(in_dim, out_dim, 'relu', 1)

    def test_linear_relu_2(self):
        in_dim, out_dim = 2, 1
        self.linear_from_params_test_helper(in_dim, out_dim, 'relu', 1)

    def test_linear_relu_3(self):
        in_dim, out_dim = 2, 2
        self.linear_from_params_test_helper(in_dim, out_dim, 'relu', 1)

    def test_linear_relu_4(self):
        in_dim, out_dim = 2, 3
        self.linear_from_params_test_helper(in_dim, out_dim, 'relu', 1)

    def test_linear_relu_5(self):
        in_dim, out_dim = 2, 3
        self.linear_from_params_test_helper(in_dim, out_dim, 'relu', 4)

    def test_linear_relu_6(self):
        in_dim, out_dim = 10, 20
        self.linear_from_params_test_helper(in_dim, out_dim, 'relu', 30)

    def test_bce_loss_1(self):
        loss = BCELoss(True)
        prediction_logits, targets = Tensor(np.array([0])), Tensor(np.array([1]))
        loss_value = loss(prediction_logits, targets)
        np.testing.assert_almost_equal(loss_value.data, np.log(2), 6)

    def test_bce_loss_2(self):
        loss = BCELoss(True)
        prediction_logits, targets = Tensor(np.array([1])), Tensor(np.array([1]))
        loss_value = loss(prediction_logits, targets)
        np.testing.assert_almost_equal(loss_value.data, np.log(1 + np.exp(-1)), 6)

    def test_bce_loss_3(self):
        loss = BCELoss(True)
        prediction_logits, targets = Tensor(np.array([-1])), Tensor(np.array([1]))
        loss_value = loss(prediction_logits, targets)
        np.testing.assert_almost_equal(loss_value.data, np.log(1 + np.exp(1)), 6)

    def test_bce_loss_4(self):
        loss = BCELoss(True)
        prediction_logits, targets = Tensor(np.array([0])), Tensor(np.array([0]))
        loss_value = loss(prediction_logits, targets)
        np.testing.assert_almost_equal(loss_value.data, np.log(2), 6)

    def test_bce_loss_5(self):
        loss = BCELoss(True)
        prediction_logits, targets = Tensor(np.array([1])), Tensor(np.array([0]))
        loss_value = loss(prediction_logits, targets)
        np.testing.assert_almost_equal(loss_value.data, np.log(1 + np.exp(1)), 6)

    def test_bce_loss_6(self):
        loss = BCELoss(True)
        prediction_logits, targets = Tensor(np.array([-1])), Tensor(np.array([0]))
        loss_value = loss(prediction_logits, targets)
        np.testing.assert_almost_equal(loss_value.data, np.log(1 + np.exp(-1)), 6)

    def test_bce_loss_7(self):
        loss = BCELoss(True)
        prediction_logits, targets = Tensor(np.array([0, 2, -2, 0, 2, -2])), Tensor(np.array([0, 0, 0, 1, 1, 1]))
        loss_value = loss(prediction_logits, targets)
        np.testing.assert_almost_equal(loss_value.data, 0.9823344, 6)

    def test_bce_loss_8(self):
        loss = BCELoss(False)
        prediction_logits, targets = Tensor(np.array([0, 2, -2, 0, 2, -2])), Tensor(np.array([0, 0, 0, 1, 1, 1]))
        loss_value = loss(prediction_logits, targets)
        gt_result = np.array([0.6931472, 2.126928, 0.12692785, 0.6931472, 0.12692805, 2.1269279])
        np.testing.assert_almost_equal(loss_value.data, gt_result, 6)

    def test_bce_loss_9(self):
        loss = BCELoss(True)
        prediction_logits, targets = Tensor(np.array([200, -9, -51, 61, 49, -55])), Tensor(np.array([0, 0, 0, 1, 1, 1]))
        loss_value = loss(prediction_logits, targets)
        np.testing.assert_almost_equal(loss_value.data, 16.666687, 6)

    def test_bce_loss_10(self):
        loss = BCELoss(False)
        prediction_logits, targets = Tensor(np.array([200, -9, -51, 61, 49, -55])), Tensor(np.array([0, 0, 0, 1, 1, 1]))
        loss_value = loss(prediction_logits, targets)
        gt_result = np.array([50, 0.000123024, 0, 0, 0, 50])
        np.testing.assert_almost_equal(loss_value.data, gt_result, 6)
