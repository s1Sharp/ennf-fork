import unittest
import numpy as np
import torch

from nn_lib import Tensor
import nn_lib.tensor_fns as F
from nn_lib.tests.utils import seeded_random
from nn_lib.mdl.layers import Conv2d
from nn_lib.mdl.loss_functions import BCELoss_logits

class TestTensorBackward(unittest.TestCase):
    """
    Tests responsible for correctness of .backward() methods of nn_lib.math_fns.Functions
    """

    #
    # property tests
    #
    def test_gradient_missing(self):
        t = Tensor(1, requires_grad=False)
        t.backward()
        self.assertIsNone(t.grad)

    def test_gradient_present(self):
        t = Tensor(1, requires_grad=True)
        t.backward()
        self.assertIsNotNone(t.grad)
        self.assertEqual(t.grad.data, 1)

    def test_gradient_zeroing(self):
        t = Tensor(1, requires_grad=True)
        t.backward()
        t.zero_grad()
        self.assertIsNotNone(t.grad)
        self.assertEqual(t.grad.data, 0)

    def test_required_grad_false(self):
        a, b = Tensor(1, requires_grad=False), Tensor(1, requires_grad=False)
        c = a + b
        self.assertFalse(c.requires_grad)

    def test_required_grad_true_1(self):
        a, b = Tensor(1, requires_grad=False), Tensor(1, requires_grad=True)
        c = a + b
        self.assertTrue(c.requires_grad)

    def test_required_grad_true_2(self):
        a, b = Tensor(1, requires_grad=True), Tensor(1, requires_grad=False)
        c = a + b
        self.assertTrue(c.requires_grad)

    def test_required_grad_true_3(self):
        a, b = Tensor(1, requires_grad=True), Tensor(1, requires_grad=True)
        c = a + b
        self.assertTrue(c.requires_grad)

    #
    # addition tests
    #
    def test_addition_gradient_scalar_1(self):
        a, b = Tensor(3, requires_grad=True), Tensor(4, requires_grad=True)
        c = a + b
        c.backward()
        self.assertEqual(a.grad.data, 1)
        self.assertEqual(b.grad.data, 1)
        self.assertEqual(c.grad.data, 1)

    def test_addition_gradient_scalar_2(self):
        a, b = Tensor(3, requires_grad=True), Tensor(4, requires_grad=True)
        c = a + b
        c.backward(Tensor(2))
        self.assertEqual(a.grad.data, 2)
        self.assertEqual(b.grad.data, 2)
        self.assertEqual(c.grad.data, 2)

    def test_addition_gradient_vector_1(self):
        a, b = Tensor([1, -2], requires_grad=True), Tensor([3, 4], requires_grad=True)
        c = a + b
        c.backward()
        np.testing.assert_equal(a.grad.data, np.array([1, 1]))
        np.testing.assert_equal(b.grad.data, np.array([1, 1]))
        np.testing.assert_equal(c.grad.data, np.array([1, 1]))

    def test_addition_gradient_vector_2(self):
        a, b = Tensor([1, -2], requires_grad=True), Tensor([3, 4], requires_grad=True)
        c = a + b
        c.backward(Tensor([-1, 5]))
        np.testing.assert_equal(a.grad.data, np.array([-1, 5]))
        np.testing.assert_equal(b.grad.data, np.array([-1, 5]))
        np.testing.assert_equal(c.grad.data, np.array([-1, 5]))

    def test_addition_gradient_mixed_1(self):
        a, b = Tensor(1, requires_grad=True), Tensor([3, 4], requires_grad=True)
        c = a + b
        c.backward(Tensor([-1, 5]))
        np.testing.assert_equal(a.grad.data, 4)
        np.testing.assert_equal(b.grad.data, np.array([-1, 5]))
        np.testing.assert_equal(c.grad.data, np.array([-1, 5]))

    def test_addition_gradient_mixed_2(self):
        a, b = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True), Tensor([3, 4], requires_grad=True)
        c = a + b
        c.backward(Tensor([[-1, 5], [0, 1], [-5, 5]]))
        np.testing.assert_equal(a.grad.data, np.array([[-1, 5], [0, 1], [-5, 5]]))
        np.testing.assert_equal(b.grad.data, np.array([-6, 11]))
        np.testing.assert_equal(c.grad.data, np.array([[-1, 5], [0, 1], [-5, 5]]))

    def test_addition_gradient_mixed_3(self):
        a, b = Tensor([[1, 5], [3, -1], [1, 0]], requires_grad=True), Tensor([[3], [4], [5]], requires_grad=True)
        c = a + b
        c.backward(Tensor([[-1, 5], [0, 1], [-4, 7]]))
        np.testing.assert_equal(a.grad.data, np.array([[-1, 5], [0, 1], [-4, 7]]))
        np.testing.assert_equal(b.grad.data, np.array([[4], [1], [3]]))
        np.testing.assert_equal(c.grad.data, np.array([[-1, 5], [0, 1], [-4, 7]]))

    #
    # negation tests
    #
    def test_negation_gradient_scalar_1(self):
        a = Tensor(5.3, requires_grad=True)
        b = -a
        b.backward()
        self.assertEqual(a.grad.data, -1)
        self.assertEqual(b.grad.data, 1)

    def test_negation_gradient_scalar_2(self):
        a = Tensor(3.4, requires_grad=True)
        b = -a
        b.backward(Tensor(2))
        self.assertEqual(a.grad.data, -2)
        self.assertEqual(b.grad.data, 2)

    def test_negation_gradient_vector_1(self):
        a = Tensor([5.3, 3.4], requires_grad=True)
        b = -a
        b.backward()
        np.testing.assert_equal(a.grad.data, np.array([-1, -1]))
        np.testing.assert_equal(b.grad.data, np.array([1, 1]))

    def test_negation_gradient_vector_2(self):
        a = Tensor([3, 14], requires_grad=True)
        b = -a
        b.backward(Tensor([1, 2]))
        np.testing.assert_equal(a.grad.data, np.array([-1, -2]))
        np.testing.assert_equal(b.grad.data, np.array([1, 2]))

    #
    # subtraction tests
    #
    def test_subtraction_gradient_scalar_1(self):
        a, b = Tensor(3, requires_grad=True), Tensor(4, requires_grad=True)
        c = a - b
        c.backward()
        self.assertEqual(a.grad.data, 1)
        self.assertEqual(b.grad.data, -1)
        self.assertEqual(c.grad.data, 1)

    def test_subtraction_gradient_scalar_2(self):
        a, b = Tensor(3.1, requires_grad=True), Tensor(4, requires_grad=True)
        c = a - b
        c.backward(Tensor(2))
        self.assertEqual(a.grad.data, 2)
        self.assertEqual(b.grad.data, -2)
        self.assertEqual(c.grad.data, 2)

    def test_subtraction_gradient_vector_1(self):
        a, b = Tensor([-1.0, -2.1], requires_grad=True), Tensor([3, 4], requires_grad=True)
        c = a - b
        c.backward()
        np.testing.assert_equal(a.grad.data, np.array([1, 1]))
        np.testing.assert_equal(b.grad.data, -np.array([1, 1]))
        np.testing.assert_equal(c.grad.data, np.array([1, 1]))

    def test_subtraction_gradient_vector_2(self):
        a, b = Tensor([11, -2], requires_grad=True), Tensor([34, 14], requires_grad=True)
        c = a - b
        c.backward(Tensor([-1.1, 5.1]))
        np.testing.assert_almost_equal(a.grad.data, np.array([-1.1, 5.1]), 6)
        np.testing.assert_almost_equal(b.grad.data, np.array([1.1, -5.1]), 6)
        np.testing.assert_almost_equal(c.grad.data, np.array([-1.1, 5.1]), 6)

    #
    # multiplication tests
    #
    def test_multiplication_gradient_scalar_1(self):
        a, b = Tensor(2, requires_grad=True), Tensor(3, requires_grad=True)
        c = a * b
        c.backward()
        self.assertEqual(a.grad.data, 3)
        self.assertEqual(b.grad.data, 2)
        self.assertEqual(c.grad.data, 1)

    def test_multiplication_gradient_scalar_2(self):
        a, b = Tensor(2, requires_grad=True), Tensor(3, requires_grad=True)
        c = a * b
        c.backward(Tensor(2))
        self.assertEqual(a.grad.data, 6)
        self.assertEqual(b.grad.data, 4)
        self.assertEqual(c.grad.data, 2)

    def test_multiplication_gradient_vector_1(self):
        a, b = Tensor([-1.5, -5.1], requires_grad=True), Tensor([3, 4], requires_grad=True)
        c = a * b
        c.backward()
        np.testing.assert_almost_equal(a.grad.data, np.array([3, 4]), 6)
        np.testing.assert_almost_equal(b.grad.data, np.array([-1.5, -5.1]), 6)
        np.testing.assert_almost_equal(c.grad.data, np.array([1, 1]), 6)

    def test_multiplication_gradient_vector_2(self):
        a, b = Tensor([5, -2], requires_grad=True), Tensor([3, 4], requires_grad=True)
        c = a * b
        c.backward(Tensor([-1.1, 5.1]))
        np.testing.assert_almost_equal(a.grad.data, np.array([-3.3, 20.4]), 6)
        np.testing.assert_almost_equal(b.grad.data, np.array([-5.5, -10.2]), 6)
        np.testing.assert_almost_equal(c.grad.data, np.array([-1.1, 5.1]), 6)

    def test_multiplication_gradient_mixed_1(self):
        a, b = Tensor(2, requires_grad=True), Tensor([-1, -4], requires_grad=True)
        c = a * b
        c.backward(Tensor([2, 4]))
        np.testing.assert_equal(a.grad.data, -18)
        np.testing.assert_equal(b.grad.data, np.array([4, 8]))
        np.testing.assert_equal(c.grad.data, np.array([2, 4]))

    def test_multiplication_gradient_mixed_2(self):
        a, b = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True), Tensor([3, 4], requires_grad=True)
        c = a * b
        c.backward()
        np.testing.assert_equal(a.grad.data, np.array([[3, 4], [3, 4], [3, 4]]))
        np.testing.assert_equal(b.grad.data, np.array([1+3+5, 2+4+6]))
        np.testing.assert_equal(c.grad.data, np.ones_like(c))

    def test_multiplication_gradient_mixed_3(self):
        a, b = Tensor([[1, 5], [3, -1], [1, 0]], requires_grad=True), Tensor([[3], [4], [5]], requires_grad=True)
        c = a * b
        c.backward()
        np.testing.assert_equal(a.grad.data, np.array([[3, 3], [4, 4], [5, 5]]))
        np.testing.assert_equal(b.grad.data, np.array([[1+5], [3-1], [1+0]]))
        np.testing.assert_equal(c.grad.data, np.ones_like(c))

    #
    # inversion tests
    #
    def test_inversion_gradient_scalar_1(self):
        a = Tensor(4, requires_grad=True)
        b = Tensor(1) / a
        b.backward()
        self.assertAlmostEqual(a.grad.data, -1/4**2)
        self.assertEqual(b.grad.data, 1)

    def test_inversion_gradient_scalar_2(self):
        a = Tensor(5, requires_grad=True)
        b = Tensor(1) / a
        b.backward(Tensor(4))
        self.assertAlmostEqual(a.grad.data, -4/5**2)
        self.assertEqual(b.grad.data, 4)

    def test_inversion_gradient_vector_1(self):
        a = Tensor([3, 5], requires_grad=True)
        b = Tensor(1) / a
        b.backward()
        np.testing.assert_almost_equal(a.grad.data, np.array([-1/9, -1/5**2]), 6)
        np.testing.assert_almost_equal(b.grad.data, np.array([1, 1]), 6)

    def test_inversion_gradient_vector_2(self):
        a = Tensor([-3, 4.5], requires_grad=True)
        b = Tensor(1) / a
        b.backward(Tensor([-1, 2]))
        np.testing.assert_almost_equal(a.grad.data, np.array([1/9, -2/4.5**2]), 6)
        np.testing.assert_almost_equal(b.grad.data, np.array([-1, 2]), 6)

    #
    # division tests
    #
    def test_division_gradient_scalar_1(self):
        a, b = Tensor(2, requires_grad=True), Tensor(3, requires_grad=True)
        c = a / b
        c.backward()
        self.assertAlmostEqual(a.grad.data, 1/3)
        self.assertAlmostEqual(b.grad.data, -2/3**2)
        self.assertEqual(c.grad.data, 1)

    def test_division_gradient_scalar_2(self):
        a, b = Tensor(-2, requires_grad=True), Tensor(5, requires_grad=True)
        c = a / b
        c.backward(Tensor(3))
        self.assertAlmostEqual(a.grad.data, 3/5)
        self.assertAlmostEqual(b.grad.data, 6/5**2)
        self.assertEqual(c.grad.data, 3)

    def test_division_gradient_vector_1(self):
        a, b = Tensor([-1.8, 3.2], requires_grad=True), Tensor([3, 4], requires_grad=True)
        c = a / b
        c.backward()
        np.testing.assert_almost_equal(a.grad.data, np.array([1/3, 1/4]), 6)
        np.testing.assert_almost_equal(b.grad.data, np.array([1.8/3**2, -3.2/4**2]), 6)
        np.testing.assert_almost_equal(c.grad.data, np.array([1, 1]), 6)

    def test_division_gradient_vector_2(self):
        a, b = Tensor([5, -2], requires_grad=True), Tensor([3.5, 1.9], requires_grad=True)
        c = a / b
        c.backward(Tensor([3.6, -1.6]))
        np.testing.assert_almost_equal(a.grad.data, np.array([3.6/3.5, -1.6/1.9]), 6)
        np.testing.assert_almost_equal(b.grad.data, np.array([-3.6*5/3.5**2, -1.6*2/1.9**2]), 6)
        np.testing.assert_almost_equal(c.grad.data, np.array([3.6, -1.6]), 6)

    def test_division_gradient_mixed_1(self):
        a, b = Tensor(2, requires_grad=True), Tensor([-1, -4], requires_grad=True)
        c = a / b
        c.backward(Tensor([2, 4]))
        np.testing.assert_equal(a.grad.data, 2/(-1) + 4/(-4))
        np.testing.assert_equal(b.grad.data, np.array([-2*2/1**2, -4*2/4**2]))
        np.testing.assert_equal(c.grad.data, np.array([2, 4]))

    def test_division_gradient_mixed_2(self):
        a, b = Tensor([-1, -4], requires_grad=True), Tensor(2, requires_grad=True)
        c = a / b
        c.backward(Tensor([3, 5]))
        np.testing.assert_equal(a.grad.data, np.array([3/2, 5/2]))
        np.testing.assert_equal(b.grad.data, 3*1/2**2 + 5*4/2**2)
        np.testing.assert_equal(c.grad.data, np.array([3, 5]))

    #
    # maximum tests
    #
    def test_maximum_scalar_1(self):
        a, b = Tensor(-1, requires_grad=True), Tensor(2, requires_grad=True)
        c = F.maximum(a, b)
        c.backward(Tensor(2))
        self.assertAlmostEqual(a.grad.data, 0)
        self.assertAlmostEqual(b.grad.data, 2)
        self.assertEqual(c.grad.data, 2)

    def test_maximum_scalar_2(self):
        a, b = Tensor(3, requires_grad=True), Tensor(2, requires_grad=True)
        c = F.maximum(a, b)
        c.backward(Tensor(-1))
        self.assertAlmostEqual(a.grad.data, -1)
        self.assertAlmostEqual(b.grad.data, 0)
        self.assertEqual(c.grad.data, -1)

    def test_maximum_scalar_3(self):
        a, b = Tensor(3, requires_grad=True), Tensor(3, requires_grad=True)
        c = F.maximum(a, b)
        c.backward(Tensor(-1))
        self.assertAlmostEqual(a.grad.data, -0.5)
        self.assertAlmostEqual(b.grad.data, -0.5)
        self.assertEqual(c.grad.data, -1)

    def test_maximum_vector_1(self):
        a, b = Tensor([3, 4], requires_grad=True), Tensor([1, 5], requires_grad=True)
        c = F.maximum(a, b)
        c.backward(Tensor([2, 3]))
        np.testing.assert_equal(a.grad.data, np.array([2, 0]))
        np.testing.assert_equal(b.grad.data, np.array([0, 3]))
        np.testing.assert_equal(c.grad.data, np.array([2, 3]))

    def test_maximum_vector_2(self):
        a, b = Tensor([3.5, 4.1, -3, 0], requires_grad=True), Tensor([3.5, 4, -2, -1], requires_grad=True)
        c = F.maximum(a, b)
        c.backward(Tensor([1, 2, 3, 4]))
        np.testing.assert_equal(a.grad.data, np.array([0.5, 2, 0, 4]))
        np.testing.assert_equal(b.grad.data, np.array([0.5, 0, 3, 0]))
        np.testing.assert_equal(c.grad.data, np.array([1, 2, 3, 4]))

    #
    # minimum tests
    #
    def test_minimum_scalar_1(self):
        a, b = Tensor(-1, requires_grad=True), Tensor(2, requires_grad=True)
        c = F.minimum(a, b)
        c.backward(Tensor(2))
        self.assertAlmostEqual(a.grad.data, 2)
        self.assertAlmostEqual(b.grad.data, 0)
        self.assertEqual(c.grad.data, 2)

    def test_minimum_scalar_2(self):
        a, b = Tensor(3, requires_grad=True), Tensor(2, requires_grad=True)
        c = F.minimum(a, b)
        c.backward(Tensor(-1))
        self.assertAlmostEqual(a.grad.data, 0)
        self.assertAlmostEqual(b.grad.data, -1)
        self.assertEqual(c.grad.data, -1)

    def test_minimum_scalar_3(self):
        a, b = Tensor(3, requires_grad=True), Tensor(3, requires_grad=True)
        c = F.minimum(a, b)
        c.backward(Tensor(-1))
        self.assertAlmostEqual(a.grad.data, -0.5)
        self.assertAlmostEqual(b.grad.data, -0.5)
        self.assertEqual(c.grad.data, -1)

    def test_minimum_vector_1(self):
        a, b = Tensor([3, 4], requires_grad=True), Tensor([1, 5], requires_grad=True)
        c = F.minimum(a, b)
        c.backward(Tensor([2, 3]))
        np.testing.assert_equal(a.grad.data, np.array([0, 3]))
        np.testing.assert_equal(b.grad.data, np.array([2, 0]))
        np.testing.assert_equal(c.grad.data, np.array([2, 3]))

    def test_minimum_vector_2(self):
        a, b = Tensor([3.5, 4.1, -3, 0], requires_grad=True), Tensor([3.5, 4, -2, -1], requires_grad=True)
        c = F.minimum(a, b)
        c.backward(Tensor([1, 2, 3, 4]))
        np.testing.assert_equal(a.grad.data, np.array([0.5, 0, 3, 0]))
        np.testing.assert_equal(b.grad.data, np.array([0.5, 2, 0, 4]))
        np.testing.assert_equal(c.grad.data, np.array([1, 2, 3, 4]))

    #
    # logarithm tests
    #
    def test_logarithm_gradient_scalar_1(self):
        a = Tensor(4, requires_grad=True)
        b = F.log(a)
        b.backward()
        self.assertAlmostEqual(a.grad.data, 1/4)
        self.assertEqual(b.grad.data, 1)

    def test_logarithm_gradient_scalar_2(self):
        a = Tensor(5, requires_grad=True)
        b = F.log(a)
        b.backward(Tensor(4))
        self.assertAlmostEqual(a.grad.data, 4/5)
        self.assertEqual(b.grad.data, 4)

    def test_logarithm_gradient_vector_1(self):
        a = Tensor([3.1, 5.4], requires_grad=True)
        b = F.log(a)
        b.backward()
        np.testing.assert_almost_equal(a.grad.data, np.array([1/3.1, 1/5.4]), 6)
        np.testing.assert_almost_equal(b.grad.data, np.array([1, 1]), 6)

    def test_logarithm_gradient_vector_2(self):
        a = Tensor([3.9, 1.5], requires_grad=True)
        b = F.log(a)
        b.backward(Tensor([1, -2.5]))
        np.testing.assert_almost_equal(a.grad.data, np.array([1/3.9, -2.5/1.5]), 6)
        np.testing.assert_almost_equal(b.grad.data, np.array([1, -2.5]), 6)

    #
    # exponent tests
    #
    def test_exponent_gradient_scalar_1(self):
        a = Tensor(4, requires_grad=True)
        b = F.exp(a)
        b.backward()
        self.assertAlmostEqual(a.grad.data, np.exp(4), 5)
        self.assertEqual(b.grad.data, 1)

    def test_exponent_gradient_scalar_2(self):
        a = Tensor(3.1, requires_grad=True)
        b = F.exp(a)
        b.backward(Tensor(4))
        self.assertAlmostEqual(a.grad.data, np.dot(4, np.exp(np.array(3.1, np.float32))), 5)
        self.assertEqual(b.grad.data, 4)

    def test_exponent_gradient_vector_1(self):
        a = Tensor([3.1, 5.4], requires_grad=True)
        b = F.exp(a)
        b.backward()
        np.testing.assert_almost_equal(a.grad.data, np.array([np.exp(3.1), np.exp(5.4)]), 4)
        np.testing.assert_almost_equal(b.grad.data, np.array([1, 1]), 6)

    def test_exponent_gradient_vector_2(self):
        a = Tensor([-3.9, 1.5], requires_grad=True)
        b = F.exp(a)
        b.backward(Tensor([1, -2.5]))
        np.testing.assert_almost_equal(a.grad.data, np.array([np.exp(-3.9), -2.5*np.exp(1.5)]), 5)
        np.testing.assert_almost_equal(b.grad.data, np.array([1, -2.5]), 6)

    #
    # matmul tests
    #
    def test_matmul_gradient_1(self):
        a, b = Tensor([[1, 2]], requires_grad=True), Tensor([[3], [4]], requires_grad=True)
        c = F.mat_mul(a, b)
        c.backward()
        np.testing.assert_almost_equal(a.grad.data, np.array([[3, 4]]))
        np.testing.assert_almost_equal(b.grad.data, np.array([[1], [2]]))
        np.testing.assert_almost_equal(c.grad.data, 1)

    def test_matmul_gradient_2(self):
        a, b = Tensor([[1, 2], [3, 4]], requires_grad=True), Tensor([[1], [-1]], requires_grad=True)
        c = F.mat_mul(a, b)
        c.backward()
        np.testing.assert_almost_equal(a.grad.data, np.array([[1, -1], [1, -1]]))
        np.testing.assert_almost_equal(b.grad.data, np.array([[4], [6]]))
        np.testing.assert_almost_equal(c.grad.data, np.array([[1], [1]]))

    def test_matmul_gradient_3(self):
        a, b = Tensor([[1, 2], [3, 4], [5, 0], [2, 1]], requires_grad=True),\
               Tensor([[1, -1, 0], [2, 3, 5]], requires_grad=True)
        c = F.mat_mul(a, b)
        c.backward()
        np.testing.assert_almost_equal(a.grad.data, np.array([[1-1+0, 2+3+5]]*4))
        np.testing.assert_almost_equal(b.grad.data, np.array([[1+3+5+2]*3, [2+4+0+1]*3]))
        np.testing.assert_almost_equal(c.grad.data, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]))

    #
    # reduction tests
    #
    def test_reduce_sum_1(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = F.reduce(a, reduction='sum')
        b.backward(Tensor(1.5))
        np.testing.assert_almost_equal(a.grad.data, np.array([1.5, 1.5, 1.5]), 5)
        np.testing.assert_almost_equal(b.grad.data, 1.5, 5)

    def test_reduce_sum_2(self):
        a = Tensor([[1, 2, 3], [3, 4, 5]], requires_grad=True)
        b = F.reduce(a, reduction='sum')
        b.backward(Tensor(2))
        np.testing.assert_almost_equal(a.grad.data, np.array([[2, 2, 2], [2, 2, 2]]), 5)
        np.testing.assert_almost_equal(b.grad.data, 2, 5)

    def test_reduce_sum_3(self):
        a = Tensor([[1.5, 2, 3], [3.5, 4, 5]], requires_grad=True)
        b = F.reduce(a, axis=0, reduction='sum')
        b.backward()
        np.testing.assert_almost_equal(a.grad.data, np.array([[1, 1, 1], [1, 1, 1]]), 5)
        np.testing.assert_almost_equal(b.grad.data, np.array([1, 1, 1]), 5)

    def test_reduce_sum_4(self):
        a_np = np.array([[[5, 2], [3, 8]], [[8, 8], [5, 0]], [[6, -1], [1, -5]]])    # (3, 2, 2)
        a = Tensor(a_np, requires_grad=True)
        b = F.reduce(a, axis=(0, -1), reduction='sum')
        b.backward()
        np.testing.assert_almost_equal(a.grad.data, np.ones_like(a_np), 5)
        np.testing.assert_almost_equal(b.grad.data, np.array([1, 1]), 5)

    def test_reduce_mean_1(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = F.reduce(a, reduction='mean')
        b.backward(Tensor(1.5))
        np.testing.assert_almost_equal(a.grad.data, np.array([1.5, 1.5, 1.5])/3, 5)
        np.testing.assert_almost_equal(b.grad.data, 1.5, 5)

    def test_reduce_mean_2(self):
        a = Tensor([[1, 2, 3], [3, 4, 5]], requires_grad=True)
        b = F.reduce(a, reduction='mean')
        b.backward(Tensor(2))
        np.testing.assert_almost_equal(a.grad.data, np.array([[2, 2, 2], [2, 2, 2]])/6, 5)
        np.testing.assert_almost_equal(b.grad.data, 2, 5)

    def test_reduce_mean_3(self):
        a = Tensor([[1.5, 2, 3], [3.5, 4, 5]], requires_grad=True)
        b = F.reduce(a, axis=0, reduction='mean')
        b.backward()
        np.testing.assert_almost_equal(a.grad.data, np.array([[1, 1, 1], [1, 1, 1]])/2, 5)
        np.testing.assert_almost_equal(b.grad.data, np.array([1, 1, 1]), 5)

    def test_reduce_mean_4(self):
        a_np = np.array([[[5, 2], [3, 8]], [[8, 8], [5, 0]], [[6, -1], [1, -5]]])    # (3, 2, 2)
        a = Tensor(a_np, requires_grad=True)
        b = F.reduce(a, axis=(0, -1), reduction='mean')
        b.backward()
        np.testing.assert_almost_equal(a.grad.data, np.ones_like(a_np)/(2*3), 5)
        np.testing.assert_almost_equal(b.grad.data, np.array([1, 1]), 5)

    #
    # slice tests
    #
    def test_slice_1(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = a[:2]
        b.backward(Tensor([3, 4]))
        np.testing.assert_almost_equal(a.grad.data, np.array([3, 4, 0]))
        np.testing.assert_almost_equal(b.grad.data, np.array([3, 4]))

    def test_slice_2(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = a[1:2]
        b.backward(Tensor(3))
        np.testing.assert_almost_equal(a.grad.data, np.array([0, 3, 0]))
        np.testing.assert_almost_equal(b.grad.data, np.array([3]))

    def test_slice_3(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = a[:]
        b.backward(Tensor([2, 4, 6]))
        np.testing.assert_almost_equal(a.grad.data, np.array([2, 4, 6]))
        np.testing.assert_almost_equal(b.grad.data, np.array([2, 4, 6]))

    def test_slice_4(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = a[1:-1]
        b.backward(Tensor(-1))
        np.testing.assert_almost_equal(a.grad.data, np.array([0, -1, 0]))
        np.testing.assert_almost_equal(b.grad.data, np.array([-1]))

    def test_slice_5(self):
        a = Tensor([[1, 2, 3]], requires_grad=True)
        b = a[0]
        b.backward(Tensor(10))
        np.testing.assert_almost_equal(a.grad.data, np.array([[10, 10, 10]]))
        np.testing.assert_almost_equal(b.grad.data, np.array([10, 10, 10]))

    def test_slice_6(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        b = a[:, 0]
        b.backward(Tensor([-1, -2]))
        np.testing.assert_almost_equal(a.grad.data, np.array([[-1, 0, 0], [-2, 0, 0]]))
        np.testing.assert_almost_equal(b.grad.data, np.array([-1, -2]))

    def test_slice_7(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        b = a[:, :2]
        b.backward(Tensor([[1, 2], [3, 4]]))
        np.testing.assert_almost_equal(a.grad.data, np.array([[1, 2, 0], [3, 4, 0]]))
        np.testing.assert_almost_equal(b.grad.data, np.array([[1, 2], [3, 4]]))

    def test_slice_8(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        b = a[1, :]
        b.backward(Tensor([5, 7, 9]))
        np.testing.assert_almost_equal(a.grad.data, np.array([[0, 0, 0], [5, 7, 9]]))
        np.testing.assert_almost_equal(b.grad.data, np.array([5, 7, 9]))

    #
    # relu tests
    #
    def test_relu_scalar_1(self):
        a = Tensor(-1, requires_grad=True)
        b = F.relu(a)
        b.backward(Tensor(2))
        self.assertAlmostEqual(a.grad.data, 0)
        self.assertAlmostEqual(b.grad.data, 2)

    def test_relu_scalar_2(self):
        a = Tensor(3, requires_grad=True)
        b = F.relu(a)
        b.backward(Tensor(-1))
        self.assertAlmostEqual(a.grad.data, -1)
        self.assertAlmostEqual(b.grad.data, -1)

    def test_relu_scalar_3(self):
        a = Tensor(0, requires_grad=True)
        b = F.relu(a)
        b.backward()
        self.assertAlmostEqual(a.grad.data, 0.5)
        self.assertAlmostEqual(b.grad.data, 1)

    def test_relu_vector_1(self):
        a = Tensor([3.5, 4.1, -3, 0], requires_grad=True)
        b = F.relu(a)
        b.backward(Tensor([1, 2, 3, 4]))
        np.testing.assert_equal(a.grad.data, np.array([1, 2, 0, 2]))
        np.testing.assert_equal(b.grad.data, np.array([1, 2, 3, 4]))

    #
    # nested ops tests
    #
    def test_nested_ops_1(self):
        def target_fn(_a, _b, _c, _d, _e):
            return (_a + _b) * (_c / (_d - _e))

        a, b, c, d, e = 13, 4.2, -7.2, 9, -1.2
        tt_a, tt_b, tt_c, tt_d, tt_e = tt_create(a), tt_create(b), tt_create(c), tt_create(d), tt_create(e)
        t_a, t_b, t_c, t_d, t_e = t_create(a), t_create(b), t_create(c), t_create(d), t_create(e)

        tt_res = target_fn(tt_a, tt_b, tt_c, tt_d, tt_e)
        t_res = target_fn(t_a, t_b, t_c, t_d, t_e)

        tt_res.backward(torch.from_numpy(np.array(-1.1)))
        t_res.backward(Tensor(-1.1))

        for t_x, tt_x in zip((t_a, t_b, t_c, t_d, t_e), (tt_a, tt_b, tt_c, tt_d, tt_e)):
            np.testing.assert_almost_equal(tt_x.grad.numpy(), t_x.grad.data, 5)

    def test_nested_ops_2(self):
        a, b, c, d, e = 5.1, [-3.1, 1.8], 3.5, -7.4, [0.9, 1.1]
        tt_a, tt_b, tt_c, tt_d, tt_e = tt_create(a), tt_create(b), tt_create(c), tt_create(d), tt_create(e)
        t_a, t_b, t_c, t_d, t_e = t_create(a), t_create(b), t_create(c), t_create(d), t_create(e)

        tt_res = torch.log(tt_a) * torch.exp(tt_b/tt_d) - tt_c + tt_e
        t_res = F.log(t_a) * F.exp(t_b/t_d) - t_c + t_e

        tt_res.backward(torch.from_numpy(np.array([0.1, 2])))
        t_res.backward(Tensor([0.1, 2]))

        for t_x, tt_x in zip((t_a, t_b, t_c, t_d, t_e), (tt_a, tt_b, tt_c, tt_d, tt_e)):
            np.testing.assert_almost_equal(tt_x.grad.numpy(), t_x.grad.data, 5)

    def test_nested_ops_3(self):
        with seeded_random(0):
            a = np.random.random((16, 5))
            b = np.random.random((5, 10))
            c = np.random.random((1, 10))
            d = np.random.random()

        tt_a, tt_b, tt_c, tt_d = tt_create(a), tt_create(b), tt_create(c), tt_create(d)
        t_a, t_b, t_c, t_d = t_create(a), t_create(b), t_create(c), t_create(d)

        tt_res = (torch.matmul(tt_a, tt_b) + tt_c) * tt_d
        t_res = (F.mat_mul(t_a, t_b) + t_c) * t_d

        tt_res.backward(torch.ones_like(tt_res) * 2)
        t_res.backward(Tensor(np.ones_like(t_res.data) * 2))

        for t_x, tt_x in zip((t_a, t_b, t_c, t_d), (tt_a, tt_b, tt_c, tt_d)):
            np.testing.assert_almost_equal(tt_x.grad.numpy(), t_x.grad.data, 5)

    def test_nested_ops_4(self):
        with seeded_random(0):
            a = np.random.random((16, 5))
            b = np.random.random((5, 10))
            c = np.random.random((1, 10))
            d = np.random.random()

        tt_a, tt_b, tt_c, tt_d = tt_create(a), tt_create(b), tt_create(c), tt_create(d)
        t_a, t_b, t_c, t_d = t_create(a), t_create(b), t_create(c), t_create(d)

        tt_res = torch.mean(torch.relu((torch.matmul(tt_a, tt_b) + tt_c) * tt_d))
        t_res = F.reduce(F.relu((F.mat_mul(t_a, t_b) + t_c) * t_d), reduction='mean')

        tt_res.backward(torch.ones_like(tt_res) * 2)
        t_res.backward(Tensor(np.ones_like(t_res.data) * 2))

        for t_x, tt_x in zip((t_a, t_b, t_c, t_d), (tt_a, tt_b, tt_c, tt_d)):
            np.testing.assert_almost_equal(tt_x.grad.numpy(), t_x.grad.data, 5)

    @unittest.skip("should be fixed by @ch7hly")
    def test_concat_1(self):
        a, b = torch.tensor(np.ones((100,128)), requires_grad=True), torch.tensor(np.ones((100,128))/2, requires_grad=True)
        c = torch.cat([a,b],1)
        d = torch.sin(c)
        d.backward()  # FIXME
        self.assertEqual(1, 1)

    @unittest.skip("should be fixed by @ch7hly")
    def test_conv_1(self):
        a = Tensor(np.random.rand(25,16,16,32), requires_grad=True)
        b = Tensor(np.random.rand(25,16,16,16), requires_grad=False)
        c = Conv2d(32,16,3,1,1)
        x = c.forward(a)
        loss = BCELoss_logits()
        l = loss.forward(x,b)
        l.backward()




def tt_create(x):
    t = torch.from_numpy(np.array(x, np.float32))
    t.requires_grad = True
    return t


def t_create(x):
    return Tensor(x, requires_grad=True)


if __name__ == '__main__':
    unittest.main()
