__author__ = 'amelie'

from copy import deepcopy

import numpy

from preimage.utils.alphabet import get_index_to_n_gram
from preimage.exceptions.n_gram import InvalidNGramLengthError, InvalidYLengthError, NoThresholdsError
from preimage.exceptions.shape import InvalidShapeError


class EulerianPath:
    """Eulerian path algorithm for the pre-image of the n-gram kernel

    Solves the pre-image of the n-gram kernel by finding an Eulerian circuit in a graph. Since the n-gram weights are
    rounded to integer values, this might predict an approximation of the exact pre-image. However, this algorithm is
    faster than the branch and bound search.

    Attributes
    ----------
    n : int
        N-gram length.
    is_merging_path : bool
        True if merges path when the graph is not connected, False otherwise (choose the longest path instead).
    min_n_gram_count : int
        The minimum number of n-gram in the string to predict.

    Notes
    -----
    This is an implementation of the eulerian path algorithm as defined in [1]_. However, there are multiple ways to
    merge the paths together when the graph is not connected. We chose to find the longest path and merge the remaining
    paths after, but this might differ from what Cortes et al. did in their work. The n-gram kernel also has a
    non-unique pre-image. For instance if the string to predict has the n-grams "ab" and "ba" both "aba" and "bab" are
    valid pre-image. In that case we chose randomly a pre-image among the possible ones (this is done by visiting the
    edges and the nodes in a random order). Note that the code could also be faster but we only cared about implementing
    the algorithm correctly.

    References
    ----------
    .. [1] Cortes, Corinna, Mehryar Mohri, and Jason Weston. "A general regression framework for learning
       string-to-string mappings." Predicting Structured Data (2007): 143-168.
    """

    def __init__(self, alphabet, n, min_length=1, is_merging_path=True):
        """Initialize the eulerian path algorithm

        Parameters
        ----------
        alphabet : list
            list of letters.
        n : int
            n-gram length.
        min_length :int
            Minimum length of the predicted string.
        is_merging_path : bool
            True if merges path when the graph is not connected, False otherwise (choose the longest path instead).
        """
        self.n = int(n)
        self.is_merging_path = is_merging_path
        self.min_n_gram_count = 1 if min_length <= self.n else min_length - self.n + 1
        self._index_to_n_gram = get_index_to_n_gram(alphabet, self.n)
        self._n_gram_count = len(alphabet) ** self.n
        self._verify_n(self.n)

    def _verify_n(self, n):
        if n <= 1:
            raise InvalidNGramLengthError(n, 1)

    def find_eulerian_path(self, n_gram_weights, y_length=None, thresholds=None):
        """Solve the pre-image of n-gram kernel.

        Rounds the n_gram_weights to integer values, creates a graph with the predicted n-grams, and predicts the output
        string by finding an eulerian path in this graph. If the graph is not connected, it will merge multiple paths
        together when is_merging_path is True, and predict the longest path otherwise.

        Parameters
        ----------
        n_gram_weights : array, shape=[len(alphabet)**n]
            Weight of each n-gram.
        y_length : int or None
            Length of the string to predict. None if thresholds are used.
        thresholds :array, shape=[len(alphabet)**n] or None
            Threshold value for each n-gram above which the n_gram_weights are rounded to one.
            None if y_length is given.

        Returns
        -------
        y: string
            The predicted string.
        """
        self._verify_weights_length_thresholds(n_gram_weights, y_length, thresholds)
        rounded_weights = self._round_weights(n_gram_weights, thresholds, y_length)
        n_gram_indexes = numpy.where(rounded_weights > 0)[0]
        n_grams = self._get_n_grams_in_selected_indexes(n_gram_indexes, rounded_weights)
        y = self._find_y_corresponding_to_n_grams(n_grams)
        if y_length is not None:
            y = y[0:y_length]
        return y

    def _verify_weights_length_thresholds(self, n_gram_weights, y_length, thresholds):
        if n_gram_weights.shape[0] != self._n_gram_count:
            raise InvalidShapeError('n_gram_weights', n_gram_weights.shape, [(self._n_gram_count,)])
        if y_length is not None and y_length < self.n:
            raise InvalidYLengthError(self.n, y_length)
        if thresholds is not None and thresholds.shape[0] != self._n_gram_count:
            raise InvalidShapeError('thresholds', thresholds.shape, [(self._n_gram_count,)])
        if thresholds is None and y_length is None:
            raise NoThresholdsError()

    def _round_weights(self, n_gram_weights, thresholds, y_length):
        if y_length is None:
            rounded_weights = self._round_weights_with_thresholds(n_gram_weights, thresholds)
        else:
            n_gram_count = y_length - self.n + 1
            rounded_weights = self._round_weights_to_n_gram_count(n_gram_weights, n_gram_count)
        return rounded_weights

    def _round_weights_to_n_gram_count(self, weights, n_gram_count_in_y):
        weights_copy = numpy.copy(weights)
        weights_copy[weights_copy < 0] = 0.
        rounded_weights = numpy.round(weights_copy)
        positive_weights = weights_copy[weights > 0]
        if rounded_weights.sum() < n_gram_count_in_y:
            rounded_weights = self._add_n_grams_to_rounded_sum(weights_copy, positive_weights, n_gram_count_in_y)
        return rounded_weights

    def _add_n_grams_to_rounded_sum(self, weights_copy, positive_weights, n_gram_count_in_y):
        while numpy.round(weights_copy).sum() < n_gram_count_in_y:
            multiplicative_factors = (numpy.ceil(positive_weights + 0.5) - 0.49) / positive_weights
            min_factor = numpy.min(multiplicative_factors[multiplicative_factors > 1.])
            weights_copy = min_factor * weights_copy
            positive_weights = min_factor * positive_weights
        return numpy.round(weights_copy)

    def _round_weights_with_thresholds(self, weights, thresholds):
        rounded_weights = numpy.asarray(weights > thresholds, dtype=numpy.int)
        non_zero_weight_count = rounded_weights.sum()
        # Avoid having zero n gram predicted
        if non_zero_weight_count < self.min_n_gram_count:
            kth_indexes = numpy.arange(0, self.min_n_gram_count)
            best_weight_indexes = numpy.argpartition(-weights, kth_indexes)[0:self.min_n_gram_count]
            best_zero_weight_indexes = best_weight_indexes[rounded_weights[best_weight_indexes] == 0]
            rounded_weights[best_zero_weight_indexes[0:self.min_n_gram_count - non_zero_weight_count]] = 1
        return rounded_weights

    def _get_n_grams_in_selected_indexes(self, selected_n_gram_indexes, rounded_weights):
        repeated_n_grams = [self._index_to_n_gram[index] for index in selected_n_gram_indexes
                            for _ in range(int(rounded_weights[index]))]
        return numpy.array(repeated_n_grams)

    def _find_y_corresponding_to_n_grams(self, n_grams):
        nodes, leaving_edges, marked_edges = self._get_nodes_and_edges(n_grams)
        if self.is_merging_path:
            path = self._merge_best_paths(nodes, leaving_edges, marked_edges, n_grams)
        else:
            path, marked_edges = self._find_best_path(nodes, leaving_edges, marked_edges, n_grams)
        y = self._transform_path_in_word(path)
        return y

    def _get_nodes_and_edges(self, n_grams):
        nodes = numpy.unique([n_gram[j:j + self.n - 1] for n_gram in n_grams for j in range(2)])
        nodes = nodes[numpy.random.permutation(nodes.shape[0])]
        random_n_grams = n_grams[numpy.random.permutation(n_grams.shape[0])]
        leaving_edges = {node: [] for node in nodes}
        marked_edges = {node: [] for node in nodes}
        self._update_leaving_and_marked_edges(leaving_edges, marked_edges, random_n_grams)
        return nodes, leaving_edges, marked_edges

    def _update_leaving_and_marked_edges(self, leaving_edges, marked_edges, random_n_grams):
        for n_gram in random_n_grams:
            leaving_edges[n_gram[0:self.n - 1]].append(n_gram[1:])
            marked_edges[n_gram[0:self.n - 1]].append(False)

    def _merge_best_paths(self, nodes, leaving_edges, marked_edges, n_grams):
        path = []
        while len(nodes) > 0:
            best_path, marked_edges = self._find_best_path(nodes, leaving_edges, marked_edges, n_grams)
            path += best_path
            nodes = [node for node, edges in marked_edges.items() if sum(edges) < len(edges)]
        return path

    def _find_best_path(self, nodes, leaving_edges, marked_edges, n_grams):
        best_path = []
        best_marked_edges = {}
        for node in nodes:
            path_marked_edges = deepcopy(marked_edges)
            path = [node] + self._euler(node, leaving_edges, path_marked_edges)
            best_path, best_marked_edges = self._update_best_path(path, path_marked_edges, best_path, best_marked_edges)
            if len(path) == len(n_grams) + 1:
                break
        return best_path, best_marked_edges

    def _euler(self, node, leaving_edges, marked_edges):
        path = []
        for edge_index, destination_node in enumerate(leaving_edges[node]):
            if not marked_edges[node][edge_index]:
                marked_edges[node][edge_index] = True
                path = [destination_node] + self._euler(destination_node, leaving_edges, marked_edges) + path
        return path

    def _update_best_path(self, path, path_marked_edges, best_path, best_marked_edges):
        if len(path) >= len(best_path):
            best_path = path
            best_marked_edges = deepcopy(path_marked_edges)
        return best_path, best_marked_edges

    def _transform_path_in_word(self, path):
        y = path[0]
        for node in path[1:]:
            y += node[-1:]
        return y