__author__ = 'amelie'

import unittest2
import numpy
import numpy.testing
from mock import patch, Mock

from preimage.models.eulerian_path_model import EulerianPathModel
from preimage.learners.structured_krr import InferenceFitParameters


class TestEulerianPathModel(unittest2.TestCase):
    def setUp(self):
        self.setup_fit_parameters()
        self.setup_predict_parameters()
        self.setup_thresholds()
        self.setup_eulerian_path_algorithm()
        self.alphabet = ['a', 'b']
        self.model_with_length = EulerianPathModel(self.alphabet, n=2, is_using_length=True)
        self.model_no_length = EulerianPathModel(self.alphabet, n=2, is_using_length=False)

    def setup_fit_parameters(self):
        weights = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        gram_matrix = numpy.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.Y_train = ['aa', 'abb', 'aa']
        self.train_lengths = [2, 3, 2]
        self.fit_parameters = InferenceFitParameters(weights, gram_matrix, Y=self.Y_train,
                                                     y_lengths=self.train_lengths)

    def setup_predict_parameters(self):
        self.Y_test = ['aa', 'baa']
        self.Y_test_weights = numpy.array([[1, 2, 3], [0, 0, 1]])
        self.Y_test_n_gram_weights = [[1, 0.1, 0, 0], [0, 0.5, 1, 0]]
        self.test_lengths = [2, 3]

    def setup_thresholds(self):
        self.aa_threshold = 0.1
        self.ab_threshold = 0.2
        self.ba_threshold = 0
        self.bb_threshold = 0.4
        self.thresholds = numpy.array([self.aa_threshold, self.ab_threshold, self.ba_threshold, self.bb_threshold])
        self.n_gram_counts = [2, 1, 0, 1]

    def setup_eulerian_path_algorithm(self):
        self.eulerian_path_mock = Mock()
        self.eulerian_path_mock.find_eulerian_path.side_effect = self.Y_test
        self.eulerian_path_patch = patch('preimage.models.eulerian_path_model.EulerianPath')
        self.eulerian_path_patch.start().return_value = self.eulerian_path_mock

    def setup_feature_space_no_length(self):
        self.n_gram_weights = [[0.5, self.ab_threshold, 0, 0.4], [self.aa_threshold, 0.3, 0, self.bb_threshold],
                               [0.4, 0, 0, 0], self.n_gram_counts, self.Y_test_n_gram_weights[0],
                               self.Y_test_n_gram_weights[1]]
        self.setup_feature_space()

    def setup_feature_space_with_length(self):
        self.n_gram_weights = self.Y_test_n_gram_weights
        self.setup_feature_space()

    def setup_feature_space(self):
        self.feature_space_mock = Mock()
        self.feature_space_mock.compute_weights.side_effect = self.n_gram_weights
        self.feature_space_patch = patch('preimage.models.eulerian_path_model.NGramFeatureSpace')
        self.feature_space_patch.start().return_value = self.feature_space_mock

    def test_model_without_length_fit_learns_correct_thresholds(self):
        self.setup_feature_space_no_length()
        self.model_no_length.fit(self.fit_parameters)

        numpy.testing.assert_array_equal(self.model_no_length._thresholds_, self.thresholds)

    def test_model_with_length_predict_returns_expected_y(self):
        self.setup_feature_space_with_length()
        self.model_with_length.fit(self.fit_parameters)

        Y = self.model_with_length.predict(self.Y_test_weights, y_lengths=self.test_lengths)

        numpy.testing.assert_array_equal(Y, self.Y_test)

    def test_model_no_length_predict_returns_expected_y(self):
        self.setup_feature_space_no_length()
        self.model_no_length.fit(self.fit_parameters)

        Y = self.model_no_length.predict(self.Y_test_weights, y_lengths=None)

        numpy.testing.assert_array_equal(Y, self.Y_test)

    def test_model_with_length_predict_sends_correct_parameters_to_feature_space(self):
        self.setup_feature_space_with_length()
        self.model_with_length.fit(self.fit_parameters)

        self.model_with_length.predict(self.Y_test_weights[0:1, 0], y_lengths=self.test_lengths)

        self.feature_space_mock.compute_weights.assert_called_with(self.Y_test_weights[0:1, 0])

    def test_model_no_length_predict_sends_correct_parameters_to_feature_space(self):
        self.setup_feature_space_no_length()
        self.model_no_length.fit(self.fit_parameters)

        self.model_no_length.predict(self.Y_test_weights[0:1, 0], y_lengths=self.test_lengths)

        self.feature_space_mock.compute_weights.assert_called_with(self.Y_test_weights[0:1, 0])

    def test_model_with_length_predict_sends_correct_parameters_to_eulerian_path_algorithm(self):
        self.setup_feature_space_with_length()
        self.model_with_length.fit(self.fit_parameters)

        self.model_with_length.predict(self.Y_test_weights[0:1], y_lengths=self.test_lengths[0:1])

        call_args = self.eulerian_path_mock.find_eulerian_path.call_args
        numpy.testing.assert_array_equal(call_args[0][0], self.Y_test_n_gram_weights[0])
        self.assertDictEqual(call_args[1], {'y_length': 2})

    def test_model_without_length_predict_sends_correct_parameters_to_eulerian_path_algorithm(self):
        self.setup_feature_space_no_length()
        self.model_no_length.fit(self.fit_parameters)

        self.model_no_length.predict(self.Y_test_weights[0:1], y_lengths=None)

        call_args = self.eulerian_path_mock.find_eulerian_path.call_args
        numpy.testing.assert_array_equal(call_args[0][0], self.Y_test_n_gram_weights[0])
        numpy.testing.assert_array_equal(call_args[1]['thresholds'], self.thresholds)


if __name__ == '__main__':
    unittest2.main()