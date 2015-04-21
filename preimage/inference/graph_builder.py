__author__ = 'amelie'

import numpy


class GraphBuilder:
    def __init__(self, alphabet, n):
        self.alphabet = alphabet
        self.n = int(n)
        self._n_gram_count = len(self.alphabet) ** self.n
        self._axis = 0 if self.n == 1 else 1
        self._entering_edges = self._get_entering_edge_indexes(self._n_gram_count, alphabet, n)

    def _get_entering_edge_indexes(self, n_gram_count, alphabet, n):
        if n == 1:
            entering_edges = numpy.arange(0, len(alphabet))
        else:
            step_size = len(self.alphabet) ** (n - 1)
            entering_edges = [numpy.tile(numpy.arange(i, n_gram_count, step_size), len(alphabet))
                              for i in range(step_size)]
            entering_edges = numpy.array(entering_edges).reshape(n_gram_count, len(alphabet))
        return entering_edges

    def build_graph(self, Graph_weights, y_length):
        n_partitions = y_length - self.n + 1
        Graph = numpy.empty((n_partitions, self._n_gram_count))
        if Graph_weights.ndim == 1:
            self._build_graph_same_weights(n_partitions, Graph, Graph_weights)
        else:
            self._build_graph_different_weights(n_partitions, Graph, Graph_weights)
        return Graph

    def _build_graph_same_weights(self, n_partitions, Graph, graph_weights):
        Graph[0, :] = numpy.copy(graph_weights)
        for i in range(1, n_partitions):
            Graph[i, :] = numpy.max(Graph[i - 1, self._entering_edges], axis=self._axis) + graph_weights

    def _build_graph_different_weights(self, n_partitions, Graph, Graph_weights):
        Graph[0, :] = numpy.copy(Graph_weights[0, :])
        for i in range(1, n_partitions):
            Graph[i, :] = numpy.max(Graph[i - 1, self._entering_edges], axis=self._axis) + Graph_weights[i, :]