__author__ = 'amelie'

import unittest2
import numpy.testing

from preimage.kernels.polynomial import PolynomialKernel


class TestPolynomialKernel(unittest2.TestCase):
    def setUp(self):
        self.X_one = [[1, 2]]
        self.X_two = [[1, 0], [1, 3]]
        self.Gram_matrix_degree_one_x_one_x_one = [[5.]]
        self.Gram_matrix_degree_two_x_one_x_one = [[25.]]
        self.Gram_matrix_degree_one_bias_one_x_one_x_one = [[6.]]
        self.Gram_matrix_normalized_x_one_x_one = [[1.]]
        self.Gram_matrix_normalized_x_one_x_two = [[0.447213595, 0.989949494]]

    def test_degree_one_x_one_x_one_polynomial_kernel_returns_expected_value(self):
        kernel = PolynomialKernel(degree=1, bias=0, is_normalized=False)
        Gram_matrix = kernel(self.X_one, self.X_one)
        numpy.testing.assert_array_equal(Gram_matrix, self.Gram_matrix_degree_one_x_one_x_one)

    def test_degree_two_x_one_x_one_polynomial_kernel_returns_expected_value(self):
        kernel = PolynomialKernel(degree=2, bias=0, is_normalized=False)
        Gram_matrix = kernel(self.X_one, self.X_one)
        numpy.testing.assert_array_equal(Gram_matrix, self.Gram_matrix_degree_two_x_one_x_one)

    def test_degree_one_bias_one_x_one_x_one_polynomial_kernel_returns_expected_value(self):
        kernel = PolynomialKernel(degree=1, bias=1, is_normalized=False)
        Gram_matrix = kernel(self.X_one, self.X_one)
        numpy.testing.assert_array_equal(Gram_matrix, self.Gram_matrix_degree_one_bias_one_x_one_x_one)

    def test_x_one_x_one_normalized_polynomial_kernel_returns_expected_value(self):
        kernel = PolynomialKernel(degree=1, bias=0, is_normalized=True)
        Gram_matrix = kernel(self.X_one, self.X_one)
        numpy.testing.assert_array_equal(Gram_matrix, self.Gram_matrix_normalized_x_one_x_one)

    def test_degree_one_x_one_x_two_normalized_polynomial_kernel_returns_expected_value(self):
        kernel = PolynomialKernel(degree=1, bias=0, is_normalized=True)
        Gram_matrix = kernel(self.X_one, self.X_two)
        numpy.testing.assert_almost_equal(Gram_matrix, self.Gram_matrix_normalized_x_one_x_two)


if __name__ == '__main__':
    unittest2.main()