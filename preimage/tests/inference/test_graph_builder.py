__author__ = 'amelie'

import unittest2
import numpy.testing

from preimage.inference.graph_builder import GraphBuilder


class TestGraphBuilder(unittest2.TestCase):
    def setUp(self):
        self.alphabet = ['a', 'b', 'c']
        self.two_letter_alphabet = ['a', 'b']
        self.weights_one_gram = numpy.array([1, 2, 3])
        self.weights_two_gram = numpy.arange(9)
        self.Weights_one_gram_length_one = numpy.array([[1, 2, 3]])
        self.Weights_one_gram_length_two = numpy.array([[1, 2, 3], [3, 2, 0]])
        self.Weights_two_gram_length_two = numpy.array([numpy.arange(9)])
        self.Weights_two_gram_length_three = numpy.array([numpy.arange(8, -1, -1), numpy.arange(9)])
        self.Weights_three_gram_length_four = numpy.array([numpy.arange(8), numpy.zeros(8)])

        self.Graph_one_gram_two_partitions = numpy.array([[1, 2, 3], [4, 5, 6]])
        self.Graph_one_gram_three_partitions = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.Graph_one_gram_two_partitions_2d_weights = numpy.array([[1, 2, 3], [6, 5, 3]])
        self.Graph_two_gram_two_partitions = numpy.array([numpy.arange(9), [6, 7, 8, 10, 11, 12, 14, 15, 16]])
        self.Graph_two_gram_two_partitions_2d_weights = numpy.array([numpy.arange(8, -1, -1),
                                                                     [8, 9, 10, 10, 11, 12, 12, 13, 14]])
        self.Graph_three_gram_two_partitions_2d_weights = numpy.array([numpy.arange(8), [4, 4, 5, 5, 6, 6, 7, 7]])

        self.y_max_one_gram_length_one = 'c'
        self.y_max_two_gram_length_two = 'cc'
        self.y_max_one_gram_length_two = 'ca'
        self.y_max_one_gram_length_three_same_weight = 'ccc'
        self.y_max_two_gram_length_three = 'acc'

    def test_one_gram_length_one_build_graph_returns_graph_with_one_partition(self):
        graph_builder = GraphBuilder(self.alphabet, n=1)

        Graph = graph_builder.build_graph(self.weights_one_gram, y_length=1)

        numpy.testing.assert_array_equal(Graph, [self.weights_one_gram])

    def test_two_gram_length_two_build_graph_returns_graph_with_one_partition(self):
        graph_builder = GraphBuilder(self.alphabet, n=2)

        Graph = graph_builder.build_graph(self.weights_two_gram, y_length=2)

        numpy.testing.assert_array_equal(Graph, [self.weights_two_gram])

    def test_one_gram_length_one_2d_weights_build_graph_returns_graph_with_one_partition(self):
        graph_builder = GraphBuilder(self.alphabet, n=1)

        Graph = graph_builder.build_graph(self.Weights_one_gram_length_one, y_length=1)

        numpy.testing.assert_array_equal(Graph, self.Weights_one_gram_length_one)

    def test_two_gram_length_two_2d_weights_build_graph_returns_graph_with_one_partition(self):
        graph_builder = GraphBuilder(self.alphabet, n=2)

        Graph = graph_builder.build_graph(self.Weights_two_gram_length_two, y_length=2)

        numpy.testing.assert_array_equal(Graph, self.Weights_two_gram_length_two)

    def test_one_gram_length_two_build_graph_returns_graph_with_two_partition(self):
        graph_builder = GraphBuilder(self.alphabet, n=1)

        Graph = graph_builder.build_graph(self.weights_one_gram, y_length=2)

        numpy.testing.assert_array_equal(Graph, self.Graph_one_gram_two_partitions)

    def test_one_gram_length_three_build_graph_returns_graph_with_three_partition(self):
        graph_builder = GraphBuilder(self.alphabet, n=1)

        Graph = graph_builder.build_graph(self.weights_one_gram, y_length=3)

        numpy.testing.assert_array_equal(Graph, self.Graph_one_gram_three_partitions)

    def test_one_gram_length_two_2d_weights_build_graph_returns_graph_with_two_partition(self):
        graph_builder = GraphBuilder(self.alphabet, n=1)

        Graph = graph_builder.build_graph(self.Weights_one_gram_length_two, y_length=2)

        numpy.testing.assert_array_equal(Graph, self.Graph_one_gram_two_partitions_2d_weights)

    def test_two_gram_length_three_build_graph_returns_graph_with_two_partition(self):
        graph_builder = GraphBuilder(self.alphabet, n=2)

        Graph = graph_builder.build_graph(self.weights_two_gram, y_length=3)

        numpy.testing.assert_array_equal(Graph, self.Graph_two_gram_two_partitions)

    def test_two_gram_length_three_2d_weights_build_graph_returns_graph_with_two_partition(self):
        graph_builder = GraphBuilder(self.alphabet, n=2)

        Graph = graph_builder.build_graph(self.Weights_two_gram_length_three, y_length=3)

        numpy.testing.assert_array_equal(Graph, self.Graph_two_gram_two_partitions_2d_weights)

    def test_three_gram_length_four_2d_weights_build_graph_returns_graph_with_two_partition(self):
        graph_builder = GraphBuilder(self.two_letter_alphabet, n=3)

        Graph = graph_builder.build_graph(self.Weights_three_gram_length_four, y_length=4)

        numpy.testing.assert_array_equal(Graph, self.Graph_three_gram_two_partitions_2d_weights)

    def test_one_gram_length_one_find_max_string_returns_n_gram_with_max_weight(self):
        graph_builder = GraphBuilder(self.alphabet, n=1)

        y_max = graph_builder.find_max_string_in_graph(self.Weights_one_gram_length_one, y_length=1)

        numpy.testing.assert_array_equal(y_max, self.y_max_one_gram_length_one)

    def test_two_gram_length_two_find_max_string_returns_n_gram_with_max_weight(self):
        graph_builder = GraphBuilder(self.alphabet, n=2)

        y_max = graph_builder.find_max_string_in_graph(self.Weights_two_gram_length_two, y_length=2)

        numpy.testing.assert_array_equal(y_max, self.y_max_two_gram_length_two)

    def test_one_gram_length_two_find_max_string_returns_expected_string(self):
        graph_builder = GraphBuilder(self.alphabet, n=1)

        y_max = graph_builder.find_max_string_in_graph(self.Weights_one_gram_length_two, y_length=2)

        numpy.testing.assert_array_equal(y_max, self.y_max_one_gram_length_two)

    def test_one_gram_length_three_same_weights_find_max_string_returns_expected_string(self):
        graph_builder = GraphBuilder(self.alphabet, n=1)

        y_max = graph_builder.find_max_string_in_graph(self.weights_one_gram, y_length=3)

        numpy.testing.assert_array_equal(y_max, self.y_max_one_gram_length_three_same_weight)

    def test_two_gram_length_three_find_max_string_returns_expected_string(self):
        graph_builder = GraphBuilder(self.alphabet, n=2)

        y_max = graph_builder.find_max_string_in_graph(self.Weights_two_gram_length_three, y_length=3)

        numpy.testing.assert_array_equal(y_max, self.y_max_two_gram_length_three)


if __name__ == '__main__':
    unittest2.main()