__author__ = 'amelie'

import numpy

from preimage.utils.alphabet import get_index_to_n_gram
from preimage.exceptions.shape import InvalidShapeError
from preimage.exceptions.n_gram import InvalidYLengthError, InvalidMinLengthError


class GraphBuilder:
    """Graph builder for the pre-image of multiple string kernels.

    Solves the pre-image problem of string kernels with constant norms (Hamming, Weighted Degree). For string kernel
    where the norm is not constant, it builds a graph that can be used to compute the bounds of partial solutions in
    a branch and bound search. The graph is constructed by dynamic programming.

    Attributes
    ----------
    alphabet : list
        List of letters.
    n : int
        N-gram length.
    """
    def __init__(self, alphabet, n):
        self.alphabet = alphabet
        self.n = int(n)
        self._n_gram_count = len(self.alphabet) ** self.n
        self._entering_edges = self._get_entering_edge_indexes(self._n_gram_count, alphabet, n)
        self._index_to_n_gram = get_index_to_n_gram(alphabet, self.n)
        self._n_gram_indexes = 0 if n == 1 else numpy.arange(0, self._n_gram_count)

    def _get_entering_edge_indexes(self, n_gram_count, alphabet, n):
        if n == 1:
            entering_edges = numpy.array([numpy.arange(0, len(alphabet))])
        else:
            step_size = len(self.alphabet) ** (n - 1)
            entering_edges = [numpy.tile(numpy.arange(i, n_gram_count, step_size), len(alphabet))
                              for i in range(step_size)]
            entering_edges = numpy.array(entering_edges).reshape(n_gram_count, len(alphabet))
        return entering_edges

    def build_graph(self, graph_weights, y_length):
        """Build the graph for the bound computation of the branch and bound search.

        Parameters
        ----------
        graph_weights : array, shape=[len(alphabet)**n] or [n_partitions, len(alphabet)**n]
            Weight of each n-gram at each position, where n_partitions is the number of n_gram in y_length. If all
            positions have the same weight (n-gram kernel), the array has the shape=[len(alphabet)**n].
        y_length : int
            Length of the string to predict.

        Returns
        -------
        graph: array,  [n_partitions, len(alphabet)**n]
            Array representation of the graph. graph[i, j] represents the maximum value of a string of length i + n
            ending with the jth n-gram.
        """
        n_partitions = y_length - self.n + 1
        self._verify_graph_weights_and_y_length(graph_weights, n_partitions)
        graph = self._initialize_graph(n_partitions, graph_weights)
        self._build_graph(n_partitions, graph, graph_weights)
        return graph

    def _build_graph(self, n_partitions, graph, graph_weights):
        for i in range(1, n_partitions):
            graph[i, :] = numpy.max(graph[i - 1, self._entering_edges], axis=1) + self._get_weights(i, graph_weights)

    def find_max_string(self, graph_weights, y_length):
        """Construct the graph and find the string of maximum value.

        Solves the pre-image of string kernels with constant norms (Hamming, Weighted Degree).

        Parameters
        ----------
        graph_weights : array, shape=[len(alphabet)**n] or [n_partitions, len(alphabet)**n]
            Weight of each n-gram at each position, where n_partitions is the number of n_gram in y_length. If all
            positions have the same weight (n-gram kernel), the array has the shape=[len(alphabet)**n].
        y_length : int
            Length of the string to predict.

        Returns
        -------
        y: string
            The predicted string.
        """
        n_partitions = y_length - self.n + 1
        self._verify_graph_weights_and_y_length(graph_weights, n_partitions)
        graph = self._initialize_graph(2, graph_weights)
        predecessors = numpy.empty((n_partitions - 1, self._n_gram_count), dtype=numpy.int)
        self._build_graph_with_predecessors(n_partitions, graph, graph_weights, predecessors)
        partition_index, n_gram_index = self._get_max_string_end_indexes(graph, n_partitions)
        max_string = self._build_max_string(partition_index, n_gram_index, predecessors)
        return max_string

    def _build_graph_with_predecessors(self, n_partitions, graph, graph_weights, predecessors):
        for i in range(1, n_partitions):
            max_entering_edge_indexes = numpy.argmax(graph[0, self._entering_edges], axis=1)
            predecessors[i - 1, :] = self._entering_edges[self._n_gram_indexes, max_entering_edge_indexes]
            graph[1, :] = graph[0, predecessors[i - 1, :]] + self._get_weights(i, graph_weights)
            graph[0, :] = graph[1, :]

    def _get_max_string_end_indexes(self, graph, n_partitions):
        n_gram_index = numpy.argmax(graph[0, :])
        partition_index = n_partitions - 2
        return partition_index, n_gram_index

    def find_max_string_in_length_range(self, graph_weights, min_y_length, max_y_length, is_normalized):
        """Construct the graph and find the string of maximum value in a given length range.

        Solves the pre-image of string kernels with constant norms (Hamming, Weighted Degree) when the length of the
        string to predict is unknown.

        Parameters
        ----------
        graph_weights : array, shape=[len(alphabet)**n] or [n_partitions, len(alphabet)**n]
            Weight of each n-gram at each position, where n_partitions is the number of n_gram in y_length. If all
            positions have the same weight (n-gram kernel), the array has the shape=[len(alphabet)**n].
        min_y_length : int
            Minimum length of the string to predict.
        max_y_length : int
            Maximum length of the string to predict.
        is_normalized : bool
            True if it solves the pre-image of the normalized kernel, False otherwise.
            (They have a different optimisation problem).

        Returns
        -------
        y: string
            The predicted string.
        """
        min_partition_index, n_partitions = self._get_min_max_partition(graph_weights, max_y_length, min_y_length)
        graph = self._initialize_graph(n_partitions, graph_weights)
        predecessors = numpy.empty((n_partitions - 1, self._n_gram_count), dtype=numpy.int)
        self._build_complete_graph_with_predecessors(n_partitions, graph, graph_weights, predecessors)
        end_indexes = self._get_max_string_end_indexes_in_range(graph, min_partition_index, n_partitions, is_normalized)
        max_string = self._build_max_string(end_indexes[0], end_indexes[1], predecessors)
        return max_string

    def _get_min_max_partition(self, graph_weights, max_y_length, min_y_length):
        n_partitions = max_y_length - self.n + 1
        min_y_length = max(self.n, min_y_length)
        self._verify_graph_weights_and_y_length(graph_weights, n_partitions)
        self._verify_min_max_length(min_y_length, max_y_length)
        min_partition_index = min_y_length - self.n
        return min_partition_index, n_partitions

    def _initialize_graph(self, n_partitions, graph_weights):
        graph = numpy.empty((n_partitions, self._n_gram_count))
        graph[0, :] = self._get_weights(0, graph_weights)
        return graph

    def _build_complete_graph_with_predecessors(self, n_partitions, graph, graph_weights, predecessors):
        for i in range(1, n_partitions):
            max_entering_edge_indexes = numpy.argmax(graph[i - 1, self._entering_edges], axis=1)
            predecessors[i - 1, :] = self._entering_edges[self._n_gram_indexes, max_entering_edge_indexes]
            graph[i, :] = graph[i - 1, predecessors[i - 1, :]] + self._get_weights(i, graph_weights)

    def _get_weights(self, i, graph_weights):
        if graph_weights.ndim == 1:
            partition_weights = graph_weights
        else:
            partition_weights = graph_weights[i, :]
        return partition_weights

    def _get_max_string_end_indexes_in_range(self, graph, min_partition, n_partitions, is_normalized):
        norm = [n_gram_count for n_gram_count in range(min_partition + 1, n_partitions + 1)]
        norm = numpy.array(norm).reshape(-1, 1)
        if is_normalized:
            graph[min_partition:, :] *= 1. / numpy.sqrt(norm)
            end_indexes = numpy.unravel_index(numpy.argmax(graph[min_partition:, :]), graph[min_partition:, :].shape)
        else:
            graph[min_partition:, :] = norm - 2 * graph[min_partition:, :]
            end_indexes = numpy.unravel_index(numpy.argmin(graph[min_partition:, :]), graph[min_partition:, :].shape)
        partition_index = end_indexes[0] + min_partition - 1
        return partition_index, end_indexes[1]

    def _build_max_string(self, partition_index, n_gram_index, predecessors):
        max_string = self._index_to_n_gram[n_gram_index]
        best_index = n_gram_index
        for i in range(partition_index, -1, -1):
            best_index = predecessors[i, best_index]
            max_string = self._index_to_n_gram[best_index][0] + max_string
        return max_string

    def _verify_graph_weights_and_y_length(self, graph_weights, n_partitions):
        if n_partitions <= 0:
            raise InvalidYLengthError(self.n, n_partitions + self.n - 1)
        valid_shapes = [(self._n_gram_count,), (n_partitions, self._n_gram_count)]
        if graph_weights.shape not in valid_shapes:
            raise InvalidShapeError('graph_weights', graph_weights.shape, valid_shapes)

    def _verify_min_max_length(self, min_length, max_length):
        if min_length > max_length:
            raise InvalidMinLengthError(min_length, max_length)