__author__ = 'amelie'

import unittest2
import numpy
import numpy.testing
from mock import Mock

from preimage.learners.structured_krr import StructuredKernelRidgeRegression, InferenceFitParameters


class TestStructuredKernelRidgeRegression(unittest2.TestCase):
    def setUp(self):
        self.setup_x_and_y()
        self.setup_kernel()
        self.setup_learners()

    def setup_x_and_y(self):
        self.X_train = [[1., 0], [0.5, 1]]
        self.X_test = [[0, 0], [1, 0], [0, 1]]
        self.Y_train = ['abc', 'ba']
        self.Y_test = ['ab', 'baa', 'a']
        self.y_test_lengths = [2, 3, 1]
        self.y_train_lengths = [1, 2]

    def setup_kernel(self):
        self.gram_matrix = numpy.array([[1., 0.5], [0.5, 1.]])
        self.gram_matrix_copy = numpy.array([[1., 0.5], [0.5, 1.]])
        self.gram_matrix_inverse = [[1.33333333, -0.66666667], [-0.66666667, 1.33333333]]
        self.gram_matrix_plus_one_half_diagonal_inverse = [[0.75, -0.25], [-0.25, 0.75]]
        self.gram_matrix_x_train_x_test = [[0, 1, 0], [0, 0, 1]]
        self.Y_weights = [[0, 0], [1.33333333, -0.66666667], [-0.66666667, 1.33333333]]
        self.kernel_mock = Mock(side_effect=[self.gram_matrix, self.gram_matrix_x_train_x_test])

    def setup_learners(self):
        self.model_mock = Mock()
        self.structured_krr = StructuredKernelRidgeRegression(alpha=0, kernel=self.kernel_mock,
                                                              inference_model=self.model_mock)

    def test_alpha_zero_structured_krr_fit_weights_is_gram_matrix_inverse(self):
        self.structured_krr.fit(self.X_train, self.Y_train)

        numpy.testing.assert_almost_equal(self.structured_krr.weights_, self.gram_matrix_inverse)

    def test_alpha_one_half_structured_krr_fit_weights_is_inverse_of_gram_matrix_plus_one_half_diagonal(self):
        structured_krr = StructuredKernelRidgeRegression(alpha=0.5, kernel=self.kernel_mock,
                                                         inference_model=self.model_mock)

        structured_krr.fit(self.X_train, self.Y_train)

        numpy.testing.assert_almost_equal(structured_krr.weights_, self.gram_matrix_plus_one_half_diagonal_inverse)

    def test_structured_krr_fit_x_train_is_equal_to_x(self):
        self.structured_krr.fit(self.X_train, self.Y_train)

        numpy.testing.assert_almost_equal(self.structured_krr.X_train_, self.X_train)

    def test_structured_krr_fit_creates_correct_inference_fit_parameters(self):
        inference_parameter_mock = Mock(return_value=None)
        InferenceFitParameters.__init__ = inference_parameter_mock

        self.structured_krr.fit(self.X_train, self.Y_train, self.y_train_lengths)

        inference_parameter_mock.assert_called_once_with(self.structured_krr.weights_, self.gram_matrix, self.Y_train,
                                                         self.y_train_lengths)

    def test_structured_krr_solve_does_not_change_gram_matrix(self):
        structured_krr = StructuredKernelRidgeRegression(alpha=0.5, kernel=self.kernel_mock,
                                                         inference_model=self.model_mock)

        structured_krr.fit(self.X_train, self.Y_train, self.y_train_lengths)

        numpy.testing.assert_array_equal(self.gram_matrix, self.gram_matrix_copy)

    def test_structured_krr_fit_calls_inference_model_fit(self):
        self.structured_krr.fit(self.X_train, self.Y_train)

        self.assertTrue(self.model_mock.fit.called)

    def test_did_not_call_fit_structured_krr_predict_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.structured_krr.predict(self.X_test)

    def test_structured_krr_predict_returns_expected_predictions(self):
        self.model_mock.predict = Mock(return_value=self.Y_test)
        self.structured_krr.fit(self.X_train, self.Y_train)

        Y_predicted = self.structured_krr.predict(self.X_test)

        numpy.testing.assert_array_equal(Y_predicted, self.Y_test)

    def test_structured_krr_predict_sends_y_weights_to_inference_predict(self):
        self.structured_krr.fit(self.X_train, self.Y_train)
        self.kernel_mock = Mock(return_value=self.gram_matrix_x_train_x_test)

        self.structured_krr.predict(self.X_test)

        numpy.testing.assert_almost_equal(self.model_mock.predict.call_args[0][0], self.Y_weights)

    def test_structured_krr_predict_sends_y_lengths_to_inference_predict(self):
        self.structured_krr.fit(self.X_train, self.Y_train)
        self.kernel_mock = Mock(return_value=self.gram_matrix_x_train_x_test)

        self.structured_krr.predict(self.X_test, self.y_test_lengths)

        numpy.testing.assert_almost_equal(self.model_mock.predict.call_args[0][1], self.y_test_lengths)


if __name__ == '__main__':
    unittest2.main()