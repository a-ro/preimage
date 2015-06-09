__author__ = 'amelie'

import unittest2
import numpy
import numpy.testing
from mock import patch, Mock

from preimage.models.string_max_model import StringMaximizationModel


class TestStringMaximizationModel(unittest2.TestCase):
    def setUp(self):
        self.setup_feature_space()
        self.setup_graph_builder()
        self.setup_branch_and_bound()
        self.alphabet = ['a', 'b', 'c']
        self.model_with_length = StringMaximizationModel(self.alphabet, n=1, gs_kernel=Mock(), max_time=30)
        self.y_weights = numpy.array([1, 0.5])
        self.Y_train = ['aa', 'bab']

    def setup_feature_space(self):
        self.n_gram_weights = numpy.array([0, 1, 0], dtype=numpy.float64)
        self.feature_space_mock = Mock()
        self.feature_space_mock.compute_weights.return_value = self.n_gram_weights
        self.feature_space_patch = patch('preimage.models.string_max_model.GenericStringSimilarityFeatureSpace')
        self.feature_space_patch.start().return_value = self.feature_space_mock

    def setup_graph_builder(self):
        self.graph = numpy.array([[0, 1, 1], [2, 3, 1]], dtype=numpy.float64)
        self.graph_builder_mock = Mock()
        self.graph_builder_mock.build_graph.return_value = self.graph
        self.graph_builder_path = patch('preimage.models.string_max_model.GraphBuilder')
        self.graph_builder_path.start().return_value = self.graph_builder_mock

    def setup_branch_and_bound(self):
        self.Y_test_length_one = ['a']
        self.bound_length_one = [1.5]
        self.Y_test_length_two = ['bb', 'aa']
        self.bounds_length_two = [3, 2]
        self.bound_factory_patch = patch('preimage.models.string_max_model.get_gs_similarity_node_creator')
        self.bound_factory_patch.start().return_value = Mock()
        self.branch_and_bound_patch = patch('preimage.models.string_max_model.branch_and_bound_multiple_solutions')

    def test_model_predict_returns_expected_y_and_bound(self):
        self.branch_and_bound_patch.start().return_value = [self.Y_test_length_one, self.bound_length_one]
        self.model_with_length.fit(self.Y_train, self.y_weights, y_length=1)

        Y, bound = self.model_with_length.predict(n_predictions=1)

        numpy.testing.assert_array_equal(Y, self.Y_test_length_one)
        numpy.testing.assert_array_equal(bound, self.bound_length_one)

    def test_two_solutions_model_predict_returns_expected_y_and_bounds(self):
        self.branch_and_bound_patch.start().return_value = [self.Y_test_length_two, self.bounds_length_two]
        self.model_with_length.fit(self.Y_train, self.y_weights, y_length=2)

        Y, bound = self.model_with_length.predict(n_predictions=2)

        numpy.testing.assert_array_equal(Y, self.Y_test_length_two)
        numpy.testing.assert_array_equal(bound, self.bounds_length_two)

    def test_model_fit_sends_correct_parameters_to_feature_space(self):
        y_length = 1
        self.branch_and_bound_patch.start().return_value = [self.Y_test_length_one, self.bound_length_one]

        self.model_with_length.fit(self.Y_train, self.y_weights, y_length)

        self.feature_space_mock.compute_weights.assert_called_with(self.y_weights, y_length)

    def test_model_fit_sends_correct_parameters_to_graph_builder(self):
        y_length = 1
        self.branch_and_bound_patch.start().return_value = [self.Y_test_length_one, self.bound_length_one]

        self.model_with_length.fit(self.Y_train, self.y_weights, y_length)

        self.graph_builder_mock.build_graph.assert_called_with(self.n_gram_weights, 1)


if __name__ == '__main__':
    unittest2.main()