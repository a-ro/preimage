__author__ = 'amelie'

import unittest2
import numpy
import numpy.testing
from mock import patch, Mock

from preimage.models.weighted_degree_model import WeightedDegreeModel
from preimage.learners.structured_krr import InferenceFitParameters
from preimage.exceptions.n_gram import NoYLengthsError


class TestWeightedDegreeModel(unittest2.TestCase):
    def setUp(self):
        self.setup_feature_space()
        self.setup_fit_parameters()
        self.setup_graph_builder()
        self.alphabet = ['a', 'b', 'c']
        self.model_with_length = WeightedDegreeModel(self.alphabet, n=2, is_using_length=True)
        self.model_no_length = WeightedDegreeModel(self.alphabet, n=2, is_using_length=False)
        self.Y_weights = [[1, 2, 3], [4, 5, 6]]
        self.y_lengths = [3, 4]

    def setup_fit_parameters(self):
        self.max_train_length = 2
        self.min_train_length = 1
        self.weights = numpy.array([[1, 2]])
        self.gram_matrix = numpy.array([[1, 0], [0, 1]])
        self.fit_parameters = InferenceFitParameters(self.weights, self.gram_matrix, Y=['a', 'ab'],
                                                     y_lengths=[1, 2])
        self.fit_parameters_no_length = InferenceFitParameters(self.weights, self.gram_matrix, Y=['a', 'ab'],
                                                               y_lengths=None)

    def setup_feature_space(self):
        self.n_gram_weights = [0, 1, 0]
        self.feature_space_mock = Mock()
        self.feature_space_mock.compute_weights.return_value = self.n_gram_weights
        self.feature_space_patch = patch('preimage.models.weighted_degree_model.WeightedDegreeFeatureSpace')
        self.feature_space_patch.start().return_value = self.feature_space_mock

    def setup_graph_builder(self):
        self.Y_test_with_length = ['aab', 'baba']
        self.Y_test_no_length = ['bb', 'aaa']
        self.graph_builder_mock = Mock()
        self.graph_builder_mock.find_max_string.side_effect = self.Y_test_with_length
        self.graph_builder_mock.find_max_string_in_length_range.side_effect = self.Y_test_no_length
        self.graph_builder_path = patch('preimage.models.weighted_degree_model.GraphBuilder')
        self.graph_builder_path.start().return_value = self.graph_builder_mock

    def test_model_with_length_fit_has_correct_min_max_lengths(self):
        self.model_with_length.fit(self.fit_parameters)

        self.assertEqual(self.model_with_length._min_length_, self.min_train_length)
        self.assertEqual(self.model_with_length._max_length_, self.max_train_length)

    def test_model_no_y_lengths_fit_has_correct_min_max_lengths(self):
        self.model_with_length.fit(self.fit_parameters_no_length)

        self.assertEqual(self.model_with_length._min_length_, self.min_train_length)
        self.assertEqual(self.model_with_length._max_length_, self.max_train_length)

    def test_no_y_lengths_model_with_length_raises_error(self):
        self.model_with_length.fit(self.fit_parameters)

        with self.assertRaises(NoYLengthsError):
            self.model_with_length.predict(self.Y_weights, y_lengths=None)

    def test_model_with_length_predict_returns_expected_y(self):
        self.model_with_length.fit(self.fit_parameters)

        Y = self.model_with_length.predict(self.Y_weights, y_lengths=self.y_lengths)

        numpy.testing.assert_array_equal(Y, self.Y_test_with_length)

    def test_model_with_length_predict_sends_correct_parameters_to_feature_space(self):
        self.model_with_length.fit(self.fit_parameters)

        self.model_with_length.predict(self.Y_weights[0:1], y_lengths=self.y_lengths[0:1])

        self.feature_space_mock.compute_weights.assert_called_with(self.Y_weights[0], self.y_lengths[0])

    def test_model_with_length_predict_sends_correct_parameters_to_graph_builder(self):
        self.model_with_length.fit(self.fit_parameters)

        self.model_with_length.predict(self.Y_weights[0:1], y_lengths=self.y_lengths[0:1])

        self.graph_builder_mock.find_max_string.assert_called_with(self.n_gram_weights, self.y_lengths[0])

    def test_model_no_length_predict_returns_expected_y(self):
        self.model_no_length.fit(self.fit_parameters)

        Y = self.model_no_length.predict(self.Y_weights, y_lengths=self.y_lengths)

        numpy.testing.assert_array_equal(Y, self.Y_test_no_length)

    def test_model_no_length_predict_sends_correct_parameters_to_feature_space(self):
        self.model_no_length.fit(self.fit_parameters)

        self.model_no_length.predict(self.Y_weights[0:1], y_lengths=None)

        self.feature_space_mock.compute_weights.assert_called_with(self.Y_weights[0], self.max_train_length)

    def test_model_no_length_predict_sends_correct_parameters_to_graph_builder(self):
        self.model_no_length.fit(self.fit_parameters)

        self.model_no_length.predict(self.Y_weights[0:1], y_lengths=None)

        self.graph_builder_mock.find_max_string_in_length_range.assert_called_with(self.n_gram_weights,
                                                                                   self.min_train_length,
                                                                                   self.max_train_length, True)


if __name__ == '__main__':
    unittest2.main()