import unittest
import numpy as np

from nn_lib import Tensor
from nn_lib.optim import SGD
from nn_lib.tests.utils import assert_almost_equal


class TestOptimizers(unittest.TestCase):
    """
    Test optimizer logic (currently SGD only)
    """

    def test_sgd_1(self):
        p = Tensor(1, requires_grad=True)
        p.grad = Tensor(1)
        optimizer = SGD([p], 0, 0)
        optimizer.step()
        assert_almost_equal(p.data, 1)

    def test_sgd_2(self):
        p = Tensor(1, requires_grad=True)
        p.grad = Tensor(1)
        optimizer = SGD([p], 1e-3, 0)
        optimizer.step()
        assert_almost_equal(p.data, 0.999)

    def test_sgd_3(self):
        p = Tensor(1, requires_grad=True)
        p.grad = Tensor(1)
        optimizer = SGD([p], 0, 5e-4)
        optimizer.step()
        assert_almost_equal(p.data, 1)

    def test_sgd_4(self):
        p = Tensor(1, requires_grad=True)
        p.grad = Tensor(1)
        optimizer = SGD([p], 1e-3, 5e-4)
        optimizer.step()
        assert_almost_equal(p.data, 0.9989995)

    def test_sgd_5(self):
        p = Tensor(2, requires_grad=True)
        p.grad = Tensor(1)
        optimizer = SGD([p], 0, 0)
        optimizer.step()
        assert_almost_equal(p.data, 2)

    def test_sgd_6(self):
        p = Tensor(2, requires_grad=True)
        p.grad = Tensor(1)
        optimizer = SGD([p], 1e-3, 0)
        optimizer.step()
        assert_almost_equal(p.data, 1.999)

    def test_sgd_7(self):
        p = Tensor(2, requires_grad=True)
        p.grad = Tensor(1)
        optimizer = SGD([p], 0, 5e-4)
        optimizer.step()
        assert_almost_equal(p.data, 2)

    def test_sgd_8(self):
        p = Tensor(2, requires_grad=True)
        p.grad = Tensor(1)
        optimizer = SGD([p], 1e-3, 5e-4)
        optimizer.step()
        assert_almost_equal(p.data, 1.998999)

    def test_sgd_9(self):
        p = Tensor(2, requires_grad=True)
        p.grad = Tensor(4)
        optimizer = SGD([p], 0, 0)
        optimizer.step()
        assert_almost_equal(p.data, 2)

    def test_sgd_10(self):
        p = Tensor(2, requires_grad=True)
        p.grad = Tensor(4)
        optimizer = SGD([p], 1e-3, 0)
        optimizer.step()
        assert_almost_equal(p.data, 1.996)

    def test_sgd_11(self):
        p = Tensor(2, requires_grad=True)
        p.grad = Tensor(4)
        optimizer = SGD([p], 0, 5e-4)
        optimizer.step()
        assert_almost_equal(p.data, 2)

    def test_sgd_12(self):
        p = Tensor(2, requires_grad=True)
        p.grad = Tensor(4)
        optimizer = SGD([p], 1e-3, 5e-4)
        optimizer.step()
        assert_almost_equal(p.data, 1.995999)

    def test_sgd_13(self):
        p = Tensor([1, 2], requires_grad=True)
        p.grad = Tensor([4, 8])
        optimizer = SGD([p], 0, 0)
        optimizer.step()
        assert_almost_equal(p.data, np.array([1, 2]))

    def test_sgd_14(self):
        p = Tensor([1, 2], requires_grad=True)
        p.grad = Tensor([4, 8])
        optimizer = SGD([p], 1e-3, 0)
        optimizer.step()
        assert_almost_equal(p.data, np.array([0.996, 1.992]))

    def test_sgd_15(self):
        p = Tensor([1, 2], requires_grad=True)
        p.grad = Tensor([4, 8])
        optimizer = SGD([p], 0, 5e-4)
        optimizer.step()
        assert_almost_equal(p.data, np.array([1, 2]))

    def test_sgd_16(self):
        p = Tensor([1, 2], requires_grad=True)
        p.grad = Tensor([4, 8])
        optimizer = SGD([p], 1e-3, 5e-4)
        optimizer.step()
        assert_almost_equal(p.data, np.array([0.9959995, 1.991999]))
