__author__ = 'amelie'

import unittest2
import numpy
import numpy.testing
from mock import patch, Mock

from preimage.models.n_gram_model import NGramModel
from preimage.learners.structured_krr import InferenceFitParameters


def branch_and_bound_side_effect(node_creator, y_length, alphabet, max_time):
    solution_dict = {1: ('a', 1), 2: ('ba', 3)}
    return solution_dict[y_length]


class TestNGramModel(unittest2.TestCase):
    def setUp(self):
        self.setup_feature_space()
        self.setup_fit_parameters()
        self.setup_graph_builder()
        self.setup_branch_and_bound()
        self.alphabet = ['a', 'b', 'c']
        self.model_with_length = NGramModel(self.alphabet, n=1, is_using_length=True)
        self.model_no_length = NGramModel(self.alphabet, n=1, is_using_length=False)
        self.Y_weights = numpy.array([[1], [0]])
        self.y_lengths = [1, 2]

    def setup_fit_parameters(self):
        self.max_train_length = 2
        self.min_train_length = 1
        self.weights = numpy.array([[1, 2]])
        self.gram_matrix = numpy.array([[1, 0], [0, 1]])
        self.fit_parameters = InferenceFitParameters(self.weights, self.gram_matrix, Y=['a', 'ab'],
                                                     y_lengths=[1, 2])

    def setup_feature_space(self):
        self.n_gram_weights = numpy.array([0, 1, 0], dtype=numpy.float64)
        self.feature_space_mock = Mock()
        self.feature_space_mock.compute_weights.return_value = self.n_gram_weights
        self.feature_space_patch = patch('preimage.models.n_gram_model.NGramFeatureSpace')
        self.feature_space_patch.start().return_value = self.feature_space_mock

    def setup_graph_builder(self):
        self.graph = numpy.array([[0, 1, 1], [2, 3, 1]], dtype=numpy.float64)
        self.graph_builder_mock = Mock()
        self.graph_builder_mock.build_graph.return_value = self.graph
        self.graph_builder_path = patch('preimage.models.n_gram_model.GraphBuilder')
        self.graph_builder_path.start().return_value = self.graph_builder_mock

    def setup_branch_and_bound(self):
        self.Y_test_with_length = ['a', 'ba']
        self.Y_test_no_length = ['bb', 'aaa']
        self.bound_factory_patch = patch('preimage.models.n_gram_model.get_n_gram_node_creator')
        self.bound_factory_patch.start().return_value = Mock()
        self.branch_and_bound_patch = patch('preimage.models.n_gram_model.branch_and_bound')
        self.branch_and_bound_patch.start().side_effect = branch_and_bound_side_effect
        self.branch_and_bound_no_length_patch = patch('preimage.models.n_gram_model.branch_and_bound_no_length')
        self.branch_and_bound_no_length_patch.start().side_effect = [['bb', 1], ['aaa', 2]]

    def test_model_with_length_predict_returns_expected_y(self):
        self.model_with_length.fit(self.fit_parameters)

        Y = self.model_with_length.predict(self.Y_weights, y_lengths=self.y_lengths)

        numpy.testing.assert_array_equal(Y, self.Y_test_with_length)

    def test_model_with_length_predict_sends_correct_parameters_to_feature_space(self):
        self.model_with_length.fit(self.fit_parameters)

        self.model_with_length.predict(self.Y_weights[0:1], y_lengths=self.y_lengths[0:1])

        self.feature_space_mock.compute_weights.assert_called_with(self.Y_weights[0:1])

    def test_model_with_length_predict_sends_correct_parameters_to_graph_builder(self):
        self.model_with_length.fit(self.fit_parameters)

        self.model_with_length.predict(self.Y_weights[0:1], y_lengths=self.y_lengths[0:1])

        self.graph_builder_mock.build_graph.assert_called_with(self.n_gram_weights, self.y_lengths[0])

    def test_model_no_length_predict_returns_expected_y(self):
        self.model_no_length.fit(self.fit_parameters)

        Y = self.model_no_length.predict(self.Y_weights, y_lengths=self.y_lengths)

        numpy.testing.assert_array_equal(Y, self.Y_test_no_length)

    def test_model_no_length_predict_sends_correct_parameters_to_feature_space(self):
        self.model_no_length.fit(self.fit_parameters)

        self.model_no_length.predict(self.Y_weights[0:1], y_lengths=self.y_lengths[0:1])

        self.feature_space_mock.compute_weights.assert_called_with(self.Y_weights[0:1])

    def test_model_no_length_predict_sends_correct_parameters_to_graph_builder(self):
        self.model_no_length.fit(self.fit_parameters)

        self.model_no_length.predict(self.Y_weights[0:1], y_lengths=self.y_lengths[0:1])

        self.graph_builder_mock.build_graph.assert_called_with(self.n_gram_weights, self.max_train_length)


if __name__ == '__main__':
    unittest2.main()