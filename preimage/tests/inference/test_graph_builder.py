__author__ = 'amelie'

import unittest2
import numpy.testing
from mock import patch

from preimage.inference.graph_builder import GraphBuilder
from preimage.exceptions.shape import InvalidShapeError
from preimage.exceptions.n_gram import InvalidYLengthError, InvalidMinLengthError


class TestGraphBuilder(unittest2.TestCase):
    def setUp(self):
        self.setup_strings()
        self.setup_weights()
        self.setup_graph()

    def setup_strings(self):
        self.alphabet = ['a', 'b', 'c']
        self.two_letter_alphabet = ['a', 'b']
        self.y_max_one_gram_length_one = 'c'
        self.y_max_two_gram_length_two = 'cc'
        self.y_max_one_gram_length_two = 'ca'
        self.y_max_one_gram_length_three = 'caa'
        self.y_max_one_gram_length_three_same_weight = 'ccc'
        self.y_max_two_gram_length_three = 'acc'
        self.index_to_n_gram_patch = patch('preimage.inference.graph_builder.get_index_to_n_gram')
        self.index_to_one_gram = {0: 'a', 1: 'b', 2: 'c'}
        self.index_to_two_gram = {0: 'aa', 1: 'ab', 2: 'ac', 3: 'ba', 4: 'bb', 5: 'bc', 6: 'ca', 7: 'cb', 8: 'cc'}

    def setup_weights(self):
        self.weights_one_gram = numpy.array([1, 2, 3])
        self.weights_two_gram = numpy.arange(9)
        self.weights_matrix_one_gram_length_one = numpy.array([[1, 2, 3]])
        self.weights_matrix_one_gram_length_two = numpy.array([[1, 2, 3], [3, 2, 0]])
        self.weights_matrix_small_last_weights = numpy.array([[1, 2, 3], [3, 2, 0], [1.3, 1, 1]])
        self.weights_matrix_big_last_weights = numpy.array([[1, 2, 3], [3, 2, 0], [1.5, 1, 1]])
        self.weights_matrix_one_gram_length_two_zero_weights = numpy.array([[1, 2, 3], [0, 0, 0]])
        self.weights_matrix_two_gram_length_two = numpy.array([numpy.arange(9)])
        self.weights_matrix_two_gram_length_three = numpy.array([numpy.arange(8, -1, -1), numpy.arange(9)])
        self.weights_matrix_three_gram_length_four = numpy.array([numpy.arange(8), numpy.zeros(8)])

    def setup_graph(self):
        self.graph_one_gram_two_partitions = numpy.array([[1, 2, 3], [4, 5, 6]])
        self.graph_one_gram_three_partitions = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.graph_one_gram_two_partitions_2d_weights = numpy.array([[1, 2, 3], [6, 5, 3]])
        self.graph_two_gram_two_partitions = numpy.array([numpy.arange(9), [6, 7, 8, 10, 11, 12, 14, 15, 16]])
        self.graph_two_gram_two_partitions_2d_weights = numpy.array([numpy.arange(8, -1, -1),
                                                                     [8, 9, 10, 10, 11, 12, 12, 13, 14]])
        self.graph_three_gram_two_partitions_2d_weights = numpy.array([numpy.arange(8), [4, 4, 5, 5, 6, 6, 7, 7]])

    def test_one_gram_length_one_build_graph_returns_graph_with_one_partition(self):
        graph_builder = GraphBuilder(self.alphabet, n=1)

        graph = graph_builder.build_graph(self.weights_one_gram, y_length=1)

        numpy.testing.assert_array_equal(graph, [self.weights_one_gram])

    def test_two_gram_length_two_build_graph_returns_graph_with_one_partition(self):
        graph_builder = GraphBuilder(self.alphabet, n=2)

        graph = graph_builder.build_graph(self.weights_two_gram, y_length=2)

        numpy.testing.assert_array_equal(graph, [self.weights_two_gram])

    def test_one_gram_length_one_2d_weights_build_graph_returns_graph_with_one_partition(self):
        graph_builder = GraphBuilder(self.alphabet, n=1)

        graph = graph_builder.build_graph(self.weights_matrix_one_gram_length_one, y_length=1)

        numpy.testing.assert_array_equal(graph, self.weights_matrix_one_gram_length_one)

    def test_two_gram_length_two_2d_weights_build_graph_returns_graph_with_one_partition(self):
        graph_builder = GraphBuilder(self.alphabet, n=2)

        graph = graph_builder.build_graph(self.weights_matrix_two_gram_length_two, y_length=2)

        numpy.testing.assert_array_equal(graph, self.weights_matrix_two_gram_length_two)

    def test_one_gram_length_two_build_graph_returns_graph_with_two_partition(self):
        graph_builder = GraphBuilder(self.alphabet, n=1)

        graph = graph_builder.build_graph(self.weights_one_gram, y_length=2)

        numpy.testing.assert_array_equal(graph, self.graph_one_gram_two_partitions)

    def test_one_gram_length_three_build_graph_returns_graph_with_three_partition(self):
        graph_builder = GraphBuilder(self.alphabet, n=1)

        graph = graph_builder.build_graph(self.weights_one_gram, y_length=3)

        numpy.testing.assert_array_equal(graph, self.graph_one_gram_three_partitions)

    def test_one_gram_length_two_2d_weights_build_graph_returns_graph_with_two_partition(self):
        graph_builder = GraphBuilder(self.alphabet, n=1)

        graph = graph_builder.build_graph(self.weights_matrix_one_gram_length_two, y_length=2)

        numpy.testing.assert_array_equal(graph, self.graph_one_gram_two_partitions_2d_weights)

    def test_two_gram_length_three_build_graph_returns_graph_with_two_partition(self):
        graph_builder = GraphBuilder(self.alphabet, n=2)

        graph = graph_builder.build_graph(self.weights_two_gram, y_length=3)

        numpy.testing.assert_array_equal(graph, self.graph_two_gram_two_partitions)

    def test_two_gram_length_three_2d_weights_build_graph_returns_graph_with_two_partition(self):
        graph_builder = GraphBuilder(self.alphabet, n=2)

        graph = graph_builder.build_graph(self.weights_matrix_two_gram_length_three, y_length=3)

        numpy.testing.assert_array_equal(graph, self.graph_two_gram_two_partitions_2d_weights)

    def test_three_gram_length_four_2d_weights_build_graph_returns_graph_with_two_partition(self):
        graph_builder = GraphBuilder(self.two_letter_alphabet, n=3)

        graph = graph_builder.build_graph(self.weights_matrix_three_gram_length_four, y_length=4)

        numpy.testing.assert_array_equal(graph, self.graph_three_gram_two_partitions_2d_weights)

    def test_one_gram_length_one_find_max_string_returns_n_gram_with_max_weight(self):
        self.index_to_n_gram_patch.start().return_value = self.index_to_one_gram
        graph_builder = GraphBuilder(self.alphabet, n=1)

        y_max = graph_builder.find_max_string(self.weights_matrix_one_gram_length_one, y_length=1)

        numpy.testing.assert_array_equal(y_max, self.y_max_one_gram_length_one)

    def test_two_gram_length_two_find_max_string_returns_n_gram_with_max_weight(self):
        self.index_to_n_gram_patch.start().return_value = self.index_to_two_gram
        graph_builder = GraphBuilder(self.alphabet, n=2)

        y_max = graph_builder.find_max_string(self.weights_matrix_two_gram_length_two, y_length=2)

        numpy.testing.assert_array_equal(y_max, self.y_max_two_gram_length_two)

    def test_one_gram_length_two_find_max_string_returns_expected_string(self):
        self.index_to_n_gram_patch.start().return_value = self.index_to_one_gram
        graph_builder = GraphBuilder(self.alphabet, n=1)

        y_max = graph_builder.find_max_string(self.weights_matrix_one_gram_length_two, y_length=2)

        numpy.testing.assert_array_equal(y_max, self.y_max_one_gram_length_two)

    def test_one_gram_length_three_same_weights_find_max_string_returns_expected_string(self):
        self.index_to_n_gram_patch.start().return_value = self.index_to_one_gram
        graph_builder = GraphBuilder(self.alphabet, n=1)

        y_max = graph_builder.find_max_string(self.weights_one_gram, y_length=3)

        numpy.testing.assert_array_equal(y_max, self.y_max_one_gram_length_three_same_weight)

    def test_two_gram_length_three_find_max_string_returns_expected_string(self):
        self.index_to_n_gram_patch.start().return_value = self.index_to_two_gram
        graph_builder = GraphBuilder(self.alphabet, n=2)

        y_max = graph_builder.find_max_string(self.weights_matrix_two_gram_length_three, y_length=3)

        numpy.testing.assert_array_equal(y_max, self.y_max_two_gram_length_three)

    def test_one_gram_one_length_in_range_find_max_string_returns_expected_string(self):
        self.index_to_n_gram_patch.start().return_value = self.index_to_one_gram
        graph_builder = GraphBuilder(self.alphabet, n=1)

        y_max = graph_builder.find_max_string_in_length_range(self.weights_matrix_one_gram_length_two, min_y_length=2,
                                                              max_y_length=2, is_normalized=False)

        numpy.testing.assert_array_equal(y_max, self.y_max_one_gram_length_two)

    def test_one_gram_one_length_in_range_normalized_find_max_string_returns_expected_string(self):
        self.index_to_n_gram_patch.start().return_value = self.index_to_one_gram
        graph_builder = GraphBuilder(self.alphabet, n=1)

        y_max = graph_builder.find_max_string_in_length_range(self.weights_matrix_one_gram_length_two, min_y_length=2,
                                                              max_y_length=2, is_normalized=True)

        numpy.testing.assert_array_equal(y_max, self.y_max_one_gram_length_two)

    def test_one_gram_two_lengths_in_range_find_max_string_returns_length_two_string(self):
        self.index_to_n_gram_patch.start().return_value = self.index_to_one_gram
        graph_builder = GraphBuilder(self.alphabet, n=1)

        y_max = graph_builder.find_max_string_in_length_range(self.weights_matrix_one_gram_length_two, min_y_length=1,
                                                              max_y_length=2, is_normalized=False)

        numpy.testing.assert_array_equal(y_max, self.y_max_one_gram_length_two)

    def test_one_gram_three_lengths_in_range_find_max_string_returns_length_two_string(self):
        self.index_to_n_gram_patch.start().return_value = self.index_to_one_gram
        graph_builder = GraphBuilder(self.alphabet, n=1)

        y_max = graph_builder.find_max_string_in_length_range(self.weights_matrix_small_last_weights, min_y_length=1,
                                                              max_y_length=3, is_normalized=True)

        numpy.testing.assert_array_equal(y_max, self.y_max_one_gram_length_two)

    def test_one_gram_three_lengths_in_range_find_max_string_returns_length_three_string(self):
        self.index_to_n_gram_patch.start().return_value = self.index_to_one_gram
        graph_builder = GraphBuilder(self.alphabet, n=1)

        y_max = graph_builder.find_max_string_in_length_range(self.weights_matrix_big_last_weights, min_y_length=1,
                                                              max_y_length=3, is_normalized=True)

        numpy.testing.assert_array_equal(y_max, self.y_max_one_gram_length_three)

    def test_zero_weights_for_length_two_find_max_string_without_length_returns_length_one_string(self):
        self.index_to_n_gram_patch.start().return_value = self.index_to_one_gram
        graph_builder = GraphBuilder(self.alphabet, n=1)

        y_max = graph_builder.find_max_string_in_length_range(self.weights_matrix_one_gram_length_two_zero_weights,
                                                              min_y_length=1, max_y_length=2, is_normalized=False)

        numpy.testing.assert_array_equal(y_max, self.y_max_one_gram_length_one)

    def test_normalized_zero_weights_for_length_two_find_max_string_without_length_returns_length_one_string(self):
        self.index_to_n_gram_patch.start().return_value = self.index_to_one_gram
        graph_builder = GraphBuilder(self.alphabet, n=1)

        y_max = graph_builder.find_max_string_in_length_range(self.weights_matrix_one_gram_length_two_zero_weights,
                                                              min_y_length=1, max_y_length=2, is_normalized=True)

        numpy.testing.assert_array_equal(y_max, self.y_max_one_gram_length_one)

    def test_two_gram_two_lengths_find_max_string_without_length_returns_length_three_string(self):
        self.index_to_n_gram_patch.start().return_value = self.index_to_two_gram
        graph_builder = GraphBuilder(self.alphabet, n=2)

        y_max = graph_builder.find_max_string_in_length_range(self.weights_matrix_two_gram_length_three,
                                                              min_y_length=2, max_y_length=3, is_normalized=False)

        numpy.testing.assert_array_equal(y_max, self.y_max_two_gram_length_three)

    def test_wrong_graph_weights_partition_count_build_graph_raises_shape_error(self):
        graph_builder = GraphBuilder(self.alphabet, n=1)

        with self.assertRaises(InvalidShapeError):
            graph_builder.build_graph(self.weights_matrix_one_gram_length_one, y_length=2)

    def test_wrong_graph_weights_n_gram_count_build_graph_raises_shape_error(self):
        graph_builder = GraphBuilder(self.alphabet, n=2)

        with self.assertRaises(InvalidShapeError):
            graph_builder.build_graph(self.weights_one_gram, y_length=2)

    def test_y_length_smaller_than_n_build_graph_raises_length_error(self):
        graph_builder = GraphBuilder(self.alphabet, n=2)

        with self.assertRaises(InvalidYLengthError):
            graph_builder.build_graph(self.weights_two_gram, y_length=1)

    def test_wrong_graph_weights_partition_count_find_max_string_raises_shape_error(self):
        graph_builder = GraphBuilder(self.alphabet, n=1)

        with self.assertRaises(InvalidShapeError):
            graph_builder.find_max_string(self.weights_matrix_one_gram_length_one, y_length=2)

    def test_wrong_graph_weights_n_gram_count_find_max_string_raises_shape_error(self):
        graph_builder = GraphBuilder(self.alphabet, n=2)

        with self.assertRaises(InvalidShapeError):
            graph_builder.find_max_string(self.weights_one_gram, y_length=2)

    def test_y_length_smaller_than_n_find_max_string_raises_length_error(self):
        graph_builder = GraphBuilder(self.alphabet, n=2)

        with self.assertRaises(InvalidYLengthError):
            graph_builder.find_max_string(self.weights_two_gram, y_length=1)

    def test_wrong_graph_weights_partition_count_find_max_string_in_range_raises_shape_error(self):
        graph_builder = GraphBuilder(self.alphabet, n=1)

        with self.assertRaises(InvalidShapeError):
            graph_builder.find_max_string_in_length_range(self.weights_matrix_one_gram_length_one, 2, 2, True)

    def test_wrong_graph_weights_n_gram_count_find_max_string_in_range_raises_shape_error(self):
        graph_builder = GraphBuilder(self.alphabet, n=2)

        with self.assertRaises(InvalidShapeError):
            graph_builder.find_max_string_in_length_range(self.weights_one_gram, 2, 2, True)

    def test_y_length_smaller_than_n_find_max_string_in_range_raises_length_error(self):
        graph_builder = GraphBuilder(self.alphabet, n=2)

        with self.assertRaises(InvalidYLengthError):
            graph_builder.find_max_string_in_length_range(self.weights_two_gram, 1, 1, True)

    def test_min_length_larger_than_max_length_find_max_string_in_range_raises_length_error(self):
        graph_builder = GraphBuilder(self.alphabet, n=2)

        with self.assertRaises(InvalidMinLengthError):
            graph_builder.find_max_string_in_length_range(self.weights_two_gram, 3, 2, True)


if __name__ == '__main__':
    unittest2.main()