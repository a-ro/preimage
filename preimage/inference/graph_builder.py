__author__ = 'amelie'

import numpy

from preimage.utils.alphabet import get_index_to_n_gram


class GraphBuilder:
    def __init__(self, alphabet, n):
        self._alphabet = alphabet
        self._n = int(n)
        self._n_gram_count = len(self._alphabet) ** self._n
        self._entering_edges = self._get_entering_edge_indexes(self._n_gram_count, alphabet, n)
        self._index_to_n_gram = get_index_to_n_gram(alphabet, self._n)
        self._n_gram_indexes = 0 if n == 1 else numpy.arange(0, self._n_gram_count)

    def _get_entering_edge_indexes(self, n_gram_count, alphabet, n):
        if n == 1:
            entering_edges = numpy.array([numpy.arange(0, len(alphabet))])
        else:
            step_size = len(self._alphabet) ** (n - 1)
            entering_edges = [numpy.tile(numpy.arange(i, n_gram_count, step_size), len(alphabet))
                              for i in range(step_size)]
            entering_edges = numpy.array(entering_edges).reshape(n_gram_count, len(alphabet))
        return entering_edges

    # todo add verification for graph weights shape
    def build_graph(self, Graph_weights, y_length):
        n_partitions = y_length - self._n + 1
        Graph = numpy.empty((n_partitions, self._n_gram_count))
        if Graph_weights.ndim == 1:
            self._build_graph_same_weights(n_partitions, Graph, Graph_weights)
        else:
            self._build_graph_different_weights(n_partitions, Graph, Graph_weights)
        return Graph

    def _build_graph_same_weights(self, n_partitions, Graph, graph_weights):
        Graph[0, :] = graph_weights
        for i in range(1, n_partitions):
            Graph[i, :] = numpy.max(Graph[i - 1, self._entering_edges], axis=1) + graph_weights

    def _build_graph_different_weights(self, n_partitions, Graph, Graph_weights):
        Graph[0, :] = Graph_weights[0, :]
        for i in range(1, n_partitions):
            Graph[i, :] = numpy.max(Graph[i - 1, self._entering_edges], axis=1) + Graph_weights[i, :]

    # todo add verification for graph weights shape
    def find_max_string_in_graph(self, Graph_weights, y_length):
        n_partitions = y_length - self._n + 1
        Graph = numpy.empty((2, self._n_gram_count))
        Predecessors = numpy.empty((n_partitions - 1, self._n_gram_count), dtype=numpy.int)
        Graph[0, :] = self._get_weights(0, Graph_weights)
        self._build_graph_with_predecessors(n_partitions, Graph, Graph_weights, Predecessors)
        max_string_last_index = numpy.argmax(Graph[0, :])
        max_string = self._build_max_string(n_partitions, Predecessors, max_string_last_index)
        return max_string

    def _build_graph_with_predecessors(self, n_partitions, Graph, Graph_weights, Predecessors):
        for i in range(1, n_partitions):
            max_entering_edge_indexes = numpy.argmax(Graph[0, self._entering_edges], axis=1)
            Predecessors[i - 1, :] = self._entering_edges[self._n_gram_indexes, max_entering_edge_indexes]
            Graph[1, :] = Graph[0, Predecessors[i - 1, :]] + self._get_weights(i, Graph_weights)
            Graph[0, :] = Graph[1, :]

    def _get_weights(self, i, Graph_weights):
        if Graph_weights.ndim == 1:
            graph_weights = Graph_weights
        else:
            graph_weights = Graph_weights[i, :]
        return graph_weights

    def _build_max_string(self, n_partitions, Predecessors, max_string_last_index):
        max_string = self._index_to_n_gram[max_string_last_index]
        best_index = max_string_last_index
        for i in range(n_partitions - 2, -1, -1):
            best_index = Predecessors[i, best_index]
            max_string = self._index_to_n_gram[best_index][0] + max_string
        return max_string