__author__ = 'amelie'

import unittest2
import numpy
import numpy.testing
from mock import Mock

from preimage.learners.structured_krr import StructuredKernelRidgeRegression, InferenceFitParameters


class TestStructuredKernelRidgeRegression(unittest2.TestCase):
    def setUp(self):
        self.X = [[1., 0], [0.5, 1]]
        self.Y = ['abc', 'ba']
        self.Gram_matrix = numpy.array([[1., 0.5], [0.5, 1.]])
        self.Gram_matrix_inverse = [[1.33333333, -0.66666667], [-0.66666667, 1.33333333]]
        self.Gram_matrix_plus_one_half_diagonal_inverse = [[0.75, -0.25], [-0.25, 0.75]]
        self.kernel_mock = Mock(return_value=self.Gram_matrix)
        self.model_mock = Mock()
        self.structured_krr = StructuredKernelRidgeRegression(alpha=0, kernel=self.kernel_mock,
                                                              inference_model=self.model_mock)

    def test_alpha_zero_structured_krr_fit_weights_is_gram_matrix_inverse(self):
        self.structured_krr.fit(self.X, self.Y)
        numpy.testing.assert_almost_equal(self.structured_krr.Weights_, self.Gram_matrix_inverse)

    def test_alpha_one_half_structured_krr_fit_weights_is_inverse_of_gram_matrix_plus_one_half_diagonal(self):
        structured_krr = StructuredKernelRidgeRegression(alpha=0.5, kernel=self.kernel_mock,
                                                         inference_model=self.model_mock)
        structured_krr.fit(self.X, self.Y)
        numpy.testing.assert_almost_equal(structured_krr.Weights_, self.Gram_matrix_plus_one_half_diagonal_inverse)

    def test_structured_krr_fit_x_train_is_equal_to_x(self):
        self.structured_krr.fit(self.X, self.Y)
        numpy.testing.assert_almost_equal(self.structured_krr.X_train_, self.X)

    def test_structured_krr_fit_creates_correct_inference_fit_parameters(self):
        inference_parameter_mock = Mock(return_value=None)
        InferenceFitParameters.__init__ = inference_parameter_mock
        self.structured_krr.fit(self.X, self.Y)
        inference_parameter_mock.assert_called_once_with(self.structured_krr.Weights_, self.Gram_matrix, self.Y)

    def test_structured_krr_fit_calls_inference_model_fit(self):
        self.structured_krr.fit(self.X, self.Y)
        self.assertTrue(self.model_mock.fit.called)


if __name__ == '__main__':
    unittest2.main()