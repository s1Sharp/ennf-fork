import unittest
import numpy as np

from nn_lib import Tensor
import nn_lib.tensor_fns as F
from nn_lib.tests.utils import seeded_random


class TestTensorForward(unittest.TestCase):
    """
    Tests responsible for correctness of .forward() methods of nn_lib.math_fns.Functions
    """

    #
    # creation tests
    #
    def test_creation_from_int(self):
        t = Tensor(1)
        self.assertTrue(t.data.dtype == np.float32)
        self.assertEqual(t.data, 1)

    def test_creation_from_float(self):
        t = Tensor(1.0)
        self.assertTrue(t.data.dtype == np.float32)
        self.assertEqual(t.data, 1)

    def test_creation_from_iterable(self):
        t = Tensor([1, 2, 3])
        self.assertTrue(t.data.dtype == np.float32)
        np.testing.assert_almost_equal(t.data, np.array([1, 2, 3]))

    def test_creation_from_ndarray_scalar(self):
        t = Tensor(np.array(1))
        self.assertTrue(t.data.dtype == np.float32)
        np.testing.assert_almost_equal(t.data, np.array(1))

    def test_creation_from_ndarray_vector(self):
        t = Tensor(np.array([1, 2, 3]))
        self.assertTrue(t.data.dtype == np.float32)
        np.testing.assert_almost_equal(t.data, np.array([1, 2, 3]))

    #
    # addition tests
    #
    def test_addition_scalar(self):
        a, b = Tensor(1), Tensor(2)
        c = a + b
        self.assertTrue(c.data.dtype == np.float32)
        self.assertEqual(c.data, 3)

    def test_addition_vector(self):
        a, b = Tensor([3, 4]), Tensor([1, 2])
        c = a + b
        self.assertTrue(c.data.dtype == np.float32)
        np.testing.assert_almost_equal(c.data, np.array([4, 6]))

    def test_addition_matrix(self):
        a, b = Tensor([[3], [4]]), Tensor([[1], [2]])
        c = a + b
        self.assertTrue(c.data.dtype == np.float32)
        np.testing.assert_almost_equal(c.data, np.array([[4], [6]]))

    def test_addition_mixed_1(self):
        a, b = Tensor(1), Tensor(np.array([1, 2]))
        c = a + b
        self.assertTrue(c.data.dtype == np.float32)
        np.testing.assert_almost_equal(c.data, np.array([2, 3]))

    def test_addition_mixed_2(self):
        a, b = Tensor(1), Tensor(np.array([[1], [2]]))
        c = a + b
        self.assertTrue(c.data.dtype == np.float32)
        np.testing.assert_almost_equal(c.data, np.array([[2], [3]]))

    def test_addition_mixed_3(self):
        a_np, b_np = np.array([[1, 2], [3, 4], [5, 6]]), np.array([3, 4])
        a, b = Tensor(a_np, requires_grad=True), Tensor(b_np, requires_grad=True)
        c = a + b
        self.assertTrue(c.data.dtype == np.float32)
        np.testing.assert_almost_equal(c.data, a_np + b_np)

    def test_addition_mixed_4(self):
        a_np, b_np = np.array([[1, 5], [3, -1], [1, 0]]), np.array([[3], [4], [5]])
        a, b = Tensor(a_np, requires_grad=True), Tensor(b_np, requires_grad=True)
        c = a + b
        self.assertTrue(c.data.dtype == np.float32)
        np.testing.assert_almost_equal(c.data, a_np + b_np)

    #
    # negation tests
    #
    def test_negation_scalar(self):
        a = Tensor(1)
        b = -a
        self.assertTrue(b.data.dtype == np.float32)
        self.assertEqual(b.data, -1)

    def test_negation_vector(self):
        a = Tensor([0, 1])
        b = -a
        self.assertTrue(b.data.dtype == np.float32)
        np.testing.assert_almost_equal(b.data, -np.array([0, 1]))

    #
    # subtraction tests
    #
    def test_subtraction_scalar(self):
        a, b = Tensor(1), Tensor(2)
        c = a - b
        self.assertTrue(c.data.dtype == np.float32)
        self.assertEqual(c.data, -1)

    def test_subtraction_vector(self):
        a, b = Tensor(np.array([3, 4])), Tensor((1, 2))
        c = a - b
        self.assertTrue(c.data.dtype == np.float32)
        np.testing.assert_almost_equal(c.data, np.array([2, 2]))

    def test_subtraction_mixed(self):
        a, b = Tensor(1), Tensor(np.array([1, 2]))
        c = a - b
        self.assertTrue(c.data.dtype == np.float32)
        np.testing.assert_almost_equal(c.data, np.array([0, -1]))

    #
    # multiplication tests
    #
    def test_multiplication_scalar(self):
        a, b = Tensor(1), Tensor(2)
        c = a * b
        self.assertTrue(c.data.dtype == np.float32)
        self.assertEqual(c.data, 2)

    def test_multiplication_vector(self):
        a, b = Tensor([0, 4]), Tensor([1, 2])
        c = a * b
        self.assertTrue(c.data.dtype == np.float32)
        np.testing.assert_almost_equal(c.data, np.array([0, 8]))

    def test_multiplication_mixed_1(self):
        a, b = Tensor(3), Tensor(np.array([1, 2, 3, 4, 5]))
        c = a * b
        self.assertTrue(c.data.dtype == np.float32)
        np.testing.assert_almost_equal(c.data, np.array([3, 6, 9, 12, 15]))

    def test_multiplication_mixed_2(self):
        a_np, b_np = np.array([[1, 2], [3, 4], [5, 6]]), np.array([3, 4])
        a, b = Tensor(a_np, requires_grad=True), Tensor(b_np, requires_grad=True)
        c = a * b
        self.assertTrue(c.data.dtype == np.float32)
        np.testing.assert_almost_equal(c.data, a_np * b_np)

    def test_multiplication_mixed_3(self):
        a_np, b_np = np.array([[1, 5], [3, -1], [1, 0]]), np.array([[3], [3], [5]])
        a, b = Tensor(a_np, requires_grad=True), Tensor(b_np, requires_grad=True)
        c = a * b
        self.assertTrue(c.data.dtype == np.float32)
        np.testing.assert_almost_equal(c.data, a_np * b_np)

    #
    # inversion tests
    #
    def test_inversion_scalar(self):
        a = Tensor(2)
        b = Tensor(1) / a
        self.assertTrue(b.data.dtype == np.float32)
        self.assertEqual(b.data, 0.5)

    def test_inversion_vector(self):
        a = Tensor([1, 4])
        b = Tensor(1) / a
        self.assertTrue(b.data.dtype == np.float32)
        np.testing.assert_almost_equal(b.data, np.array([1, 0.25]))

    #
    # division tests
    #
    def test_division_scalar(self):
        a, b = Tensor(-1), Tensor(2)
        c = a / b
        self.assertTrue(c.data.dtype == np.float32)
        self.assertEqual(c.data, -0.5)

    def test_division_vector(self):
        a, b = Tensor([3, 4]), Tensor([1, 2])
        c = a / b
        self.assertTrue(c.data.dtype == np.float32)
        np.testing.assert_almost_equal(c.data, np.array([3, 2]))

    def test_division_mixed(self):
        a, b = Tensor(1), Tensor(np.array([1, 2]))
        c = a / b
        self.assertTrue(c.data.dtype == np.float32)
        np.testing.assert_almost_equal(c.data, np.array([1, 0.5]))

    #
    # maximum tests
    #
    def test_maximum_scalar(self):
        a, b = Tensor(-1), Tensor(2)
        c = F.maximum(a, b)
        self.assertTrue(c.data.dtype == np.float32)
        self.assertEqual(c.data, 2)

    def test_maximum_vector(self):
        a, b = Tensor([3, 4]), Tensor([1, 5])
        c = F.maximum(a, b)
        self.assertTrue(c.data.dtype == np.float32)
        np.testing.assert_almost_equal(c.data, np.array([3, 5]))

    #
    # minimum tests
    #
    def test_minimum_scalar(self):
        a, b = Tensor(-1), Tensor(2)
        c = F.minimum(a, b)
        self.assertTrue(c.data.dtype == np.float32)
        self.assertEqual(c.data, -1)

    def test_minimum_vector(self):
        a, b = Tensor([3, 4]), Tensor([1, 5])
        c = F.minimum(a, b)
        self.assertTrue(c.data.dtype == np.float32)
        np.testing.assert_almost_equal(c.data, np.array([1, 4]))

    #
    # logarithm tests
    #
    def test_logarithm_scalar(self):
        a = Tensor(1)
        b = F.log(a)
        self.assertTrue(b.data.dtype == np.float32)
        self.assertEqual(b.data, 0)

    def test_logarithm_vector(self):
        a = Tensor([3, 4])
        b = F.log(a)
        self.assertTrue(b.data.dtype == np.float32)
        np.testing.assert_almost_equal(b.data, np.log(np.array([3, 4])))

    #
    # exponent tests
    #
    def test_exponent_scalar(self):
        a = Tensor(1)
        b = F.exp(a)
        self.assertTrue(b.data.dtype == np.float32)
        np.testing.assert_almost_equal(b.data, np.e, 5)

    def test_exponent_vector(self):
        a = Tensor([5, 0.1])
        b = F.exp(a)
        self.assertTrue(b.data.dtype == np.float32)
        np.testing.assert_almost_equal(b.data, np.exp(np.array([5, 0.1], np.float32)))

    #
    # matrix by vector multiplication tests
    #
    def test_mat_mul_1(self):
        a_np, b_np = np.array([[1, 2]]), np.array([[3], [4]])
        a, b = Tensor(a_np), Tensor(b_np)
        c = F.mat_mul(a, b)
        self.assertTrue(c.data.dtype == np.float32)
        np.testing.assert_almost_equal(c.data, 11)

    def test_mat_mul_2(self):
        a_np, b_np = np.array([[1, 2], [3, 4]]), np.array([[1], [2]])
        a, b = Tensor(a_np), Tensor(b_np)
        c = F.mat_mul(a, b)
        self.assertTrue(c.data.dtype == np.float32)
        np.testing.assert_almost_equal(c.data, np.matmul(a_np, b_np))

    def test_mat_mul_3(self):
        a_np, b_np = np.array([[-1.234, 0.2352], [-11, 423.4]]), np.array([[1.1], [0.2]])
        a, b = Tensor(a_np), Tensor(b_np)
        c = F.mat_mul(a, b)
        self.assertTrue(c.data.dtype == np.float32)
        np.testing.assert_almost_equal(c.data, np.matmul(a_np, b_np), 5)

    def test_mat_mul_4(self):
        a_np, b_np = np.array([[1, 2], [3, 4], [-1.234, 0.2352]]), np.array([[1, 2], [1.1, 0.2]])
        a, b = Tensor(a_np), Tensor(b_np)
        c = F.mat_mul(a, b)
        self.assertTrue(c.data.dtype == np.float32)
        np.testing.assert_almost_equal(c.data, np.matmul(a_np, b_np), 5)

    #
    # reduction tests
    #
    def test_reduce_sum_1(self):
        a = Tensor([1, 2, 3])
        b = F.reduce(a, reduction='sum')
        self.assertTrue(b.data.dtype == np.float32)
        np.testing.assert_almost_equal(b.data, 1+2+3, 5)

    def test_reduce_sum_2(self):
        a = Tensor([[1, 2, 3], [3, 4, 5]])
        b = F.reduce(a, reduction='sum')
        self.assertTrue(b.data.dtype == np.float32)
        np.testing.assert_almost_equal(b.data, 1+2+3 + 3+4+5, 5)

    def test_reduce_sum_3(self):
        a = Tensor([[1.5, 2, 3], [3.5, 4, 5]])
        b = F.reduce(a, axis=0, reduction='sum')
        self.assertTrue(b.data.dtype == np.float32)
        np.testing.assert_almost_equal(b.data, np.array([1.5+3.5, 2+4, 3+5]), 5)

    def test_reduce_sum_4(self):
        a_np = np.array([[[5, 2], [3, 8]], [[8, 8], [5, 0]], [[6, -1], [1, -5]]])    # (3, 2, 2)
        a = Tensor(a_np)
        b = F.reduce(a, axis=(0, -1), reduction='sum')
        self.assertTrue(b.data.dtype == np.float32)
        np.testing.assert_almost_equal(b.data, np.sum(a_np, (0, -1)), 5)

    def test_reduce_mean_1(self):
        a = Tensor([1, 2, 3])
        b = F.reduce(a, reduction='mean')
        self.assertTrue(b.data.dtype == np.float32)
        np.testing.assert_almost_equal(b.data, (1+2+3)/3, 5)

    def test_reduce_mean_2(self):
        a = Tensor([[1, 2, 3], [3, 4, 5]])
        b = F.reduce(a, reduction='mean')
        self.assertTrue(b.data.dtype == np.float32)
        np.testing.assert_almost_equal(b.data, (1+2+3 + 3+4+5) / (2 * 3), 5)

    def test_reduce_mean_3(self):
        a = Tensor([[1.5, 2, 3], [3.5, 4, 5]])
        b = F.reduce(a, axis=0, reduction='mean')
        self.assertTrue(b.data.dtype == np.float32)
        np.testing.assert_almost_equal(b.data, np.array([(1.5+3.5)/2, (2+4)/2, (3+5)/2]), 5)

    def test_reduce_mean_4(self):
        a_np = np.array([[[5, 2], [3, 8]], [[8, 8], [5, 0]], [[6, -1], [1, -5]]])    # (3, 2, 2)
        a = Tensor(a_np)
        b = F.reduce(a, axis=(0, -1), reduction='mean')
        self.assertTrue(b.data.dtype == np.float32)
        np.testing.assert_almost_equal(b.data, np.mean(a_np, (0, -1)), 5)

    #
    # slice tests
    #
    def test_slice_1(self):
        a = Tensor([1, 2, 3])
        b = a[:2]
        self.assertTrue(b.data.dtype == np.float32)
        np.testing.assert_almost_equal(b.data, np.array([1, 2]))

    def test_slice_2(self):
        a = Tensor([1, 2, 3])
        b = a[1:2]
        self.assertTrue(b.data.dtype == np.float32)
        np.testing.assert_almost_equal(b.data, np.array([2]))

    def test_slice_3(self):
        a = Tensor([1, 2, 3])
        b = a[:]
        self.assertTrue(b.data.dtype == np.float32)
        np.testing.assert_almost_equal(b.data, np.array([1, 2, 3]))

    def test_slice_4(self):
        a = Tensor([1, 2, 3])
        b = a[1:-1]
        self.assertTrue(b.data.dtype == np.float32)
        np.testing.assert_almost_equal(b.data, np.array([2]))

    def test_slice_5(self):
        a = Tensor([[1, 2, 3]])
        b = a[0]
        self.assertTrue(b.data.dtype == np.float32)
        np.testing.assert_almost_equal(b.data, np.array([1, 2, 3]))

    def test_slice_6(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]])
        b = a[:, 0]
        self.assertTrue(b.data.dtype == np.float32)
        np.testing.assert_almost_equal(b.data, np.array([1, 4]))

    def test_slice_7(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]])
        b = a[:, :2]
        self.assertTrue(b.data.dtype == np.float32)
        np.testing.assert_almost_equal(b.data, np.array([[1, 2], [4, 5]]))

    def test_slice_8(self):
        a = Tensor([[1, 2, 3], [4, 5, 6]])
        b = a[1, :]
        self.assertTrue(b.data.dtype == np.float32)
        np.testing.assert_almost_equal(b.data, np.array([4, 5, 6]))

    #
    # relu tests
    #
    def test_relu_scalar_1(self):
        a = Tensor(-1)
        b = F.relu(a)
        self.assertTrue(b.data.dtype == np.float32)
        self.assertEqual(b.data, 0)

    def test_relu_scalar_2(self):
        a = Tensor(0)
        b = F.relu(a)
        self.assertTrue(b.data.dtype == np.float32)
        self.assertEqual(b.data, 0)

    def test_relu_scalar_3(self):
        a = Tensor(2)
        b = F.relu(a)
        self.assertTrue(b.data.dtype == np.float32)
        self.assertEqual(b.data, 2)

    def test_relu_vector(self):
        a = Tensor([3, -4, 0])
        b = F.relu(a)
        self.assertTrue(b.data.dtype == np.float32)
        np.testing.assert_almost_equal(b.data, np.array([3, 0, 0]))

    #
    # nested ops tests
    #
    def test_nested_ops_1(self):
        a, b, c, d, e = 31.2, 1.3, 5.1, 8.0, -12
        result = e / ((a + b) * (-c) / d)
        a, b, c, d, e = Tensor(a), Tensor(b), Tensor(c), Tensor(d), Tensor(e)
        result_tensor = e / ((a + b) * (-c) / d)
        np.testing.assert_almost_equal(result, result_tensor.data)

    def test_nested_ops_2(self):
        a, b, c, d, e = 41.0, -13.1, 75, np.array([6.4, 5]), np.array([-2.4, 0])
        result = np.log(a) * np.exp(b/d) - c + e
        a, b, c, d, e = Tensor(a), Tensor(b), Tensor(c), Tensor(d), Tensor(e)
        result_tensor = F.log(a) * F.exp(b/d) - c + e
        np.testing.assert_almost_equal(result, result_tensor.data, 5)

    def test_nested_ops_3(self):
        with seeded_random(0):
            x = np.random.random((16, 5))
            w = np.random.random((5, 10))
            b = np.random.random((1, 10))
            s = np.random.random()
        result = (np.matmul(x, w) + b) * s
        x, w, b, s = Tensor(x), Tensor(w), Tensor(b), Tensor(s)
        result_tensor = (F.mat_mul(x, w) + b) * s
        np.testing.assert_almost_equal(result, result_tensor.data, 5)


if __name__ == '__main__':
    unittest.main()
