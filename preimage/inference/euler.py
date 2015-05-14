__author__ = 'amelie'

from copy import deepcopy

import numpy

from preimage.utils.alphabet import get_index_to_n_gram


class EulerianPath:
    def __init__(self, alphabet, n, min_length, is_merging_path=False):
        self.alphabet = alphabet
        self.n = int(n)
        self.is_merging_path = is_merging_path
        self.min_n_gram_count = int(n) if min_length is None else min_length - self.n + 1
        self.index_to_n_gram = get_index_to_n_gram(alphabet, self.n)

    def find_eulerian_path(self, N_gram_weights, y_length=None, thresholds=None):
        if y_length is None:
            rounded_weights = self._round_weights_with_thresholds(N_gram_weights, thresholds)
        else:
            n_gram_count = y_length - self.n + 1
            rounded_weights = self.round_weights_to_n_gram_count(N_gram_weights, n_gram_count)
        n_gram_indexes = numpy.where(rounded_weights > 0)[0]
        n_grams = self._get_n_grams_in_selected_indexes(n_gram_indexes, rounded_weights)
        y = self._find_y_corresponding_to_n_grams(n_grams)
        if y_length is not None:
            y = y[0:y_length]
        return y

    def _round_weights_with_thresholds(self, weights, thresholds):
        rounded_weights = numpy.asarray(weights > thresholds, dtype=numpy.int)
        non_zero_weight_count = rounded_weights.sum()
        # Avoid having zero n gram predicted
        if non_zero_weight_count < self.min_n_gram_count:
            best_weights_index = numpy.argpartition(-weights, self.min_n_gram_count)
            best_weights_index = best_weights_index[rounded_weights[best_weights_index] == 0]
            rounded_weights[best_weights_index[0:self.min_n_gram_count - non_zero_weight_count]] = 1
        return rounded_weights

    # Didn't check cases where weights are all zero, all negative, or all the same
    # The case when the rounded_n_gram_count > n_gram_count_in_y didn't happen for ocr-letter
    # In cases where some weight values are equal, the rounding might give rounded_n_gram_count > n_gram_count_in_y
    # In that case, we computed a euler tour with all the predicted n-grams and cut to get the right length after
    def round_weights_to_n_gram_count(self, weights, n_gram_count_in_y):
        weights_copy = numpy.copy(weights)
        weights_copy[weights_copy < 0] = 0.
        rounded_weights = numpy.round(weights_copy)
        rounded_n_gram_count = rounded_weights.sum()
        positive_weights = weights_copy[weights > 0]
        if rounded_n_gram_count < n_gram_count_in_y:
            rounded_weights = self._add_n_grams_to_rounded_sum(weights_copy, positive_weights, n_gram_count_in_y)
        return rounded_weights

    def _add_n_grams_to_rounded_sum(self, weights_copy, positive_weights, n_gram_count_in_y):
        while (numpy.round(weights_copy).sum() < n_gram_count_in_y):
            multiplicative_factors = (numpy.ceil(positive_weights + 0.5) - 0.49) / positive_weights
            min_factor = numpy.min(multiplicative_factors[multiplicative_factors > 1.])
            weights_copy = min_factor * weights_copy
            positive_weights = min_factor * positive_weights
        return numpy.round(weights_copy)

    def _get_n_grams_in_selected_indexes(self, selected_n_gram_indexes, rounded_weights):
        repeated_n_grams = [self.index_to_n_gram[index] for index in selected_n_gram_indexes
                            for _ in range(int(rounded_weights[index]))]
        return numpy.array(repeated_n_grams)

    def _find_y_corresponding_to_n_grams(self, n_grams):
        y = ""
        nodes, leaving_edges, marked_edges = self._get_nodes_and_leaving_edges(n_grams)
        is_first_path = True
        if self.is_merging_path:
            while len(nodes) > 0:
                best_path, marked_edges = self._find_best_path(nodes, leaving_edges, marked_edges, n_grams)
                y += self._transform_path_in_word(best_path, is_first_path)
                is_first_path = False
                nodes = [node for node, edges in marked_edges.items() if sum(edges) < len(edges)]
        else:
            best_path, marked_edges = self._find_best_path(nodes, leaving_edges, marked_edges, n_grams)
            y = self._transform_path_in_word(best_path, is_first_path)
        return y

    def _get_nodes_and_leaving_edges(self, n_grams):
        nodes = numpy.unique([n_gram[j:j + self.n - 1] for n_gram in n_grams for j in range(2)])
        permutations = numpy.random.permutation(nodes.shape[0])
        nodes = nodes[permutations]
        permutations = numpy.random.permutation(n_grams.shape[0])
        random_n_grams = deepcopy(n_grams)[permutations]
        leaving_edges = {node: [] for node in nodes}
        marked_edges = {node: [] for node in nodes}
        for n_gram in random_n_grams:
            leaving_edges[n_gram[0:self.n - 1]].append(n_gram[1:])
            marked_edges[n_gram[0:self.n - 1]].append(False)
        return nodes, leaving_edges, marked_edges

    def _euler(self, node, leaving_edges, marked_edges):
        path = []
        for edge_index, destination_node in enumerate(leaving_edges[node]):
            if not marked_edges[node][edge_index]:
                marked_edges[node][edge_index] = True
                old_path = deepcopy(path)
                path = [destination_node] + self._euler(destination_node, leaving_edges, marked_edges) + old_path
        return path

    def _find_best_path(self, nodes, leaving_edges, marked_edges, n_grams):
        best_path = []
        best_path_marked_edges = {}
        best_path_length = 0
        for node in nodes:
            path_marked_edges = deepcopy(marked_edges)
            path = [node] + self._euler(node, leaving_edges, path_marked_edges)
            if len(path) >= best_path_length:
                best_path = path
                best_path_marked_edges = deepcopy(path_marked_edges)
                if len(path) == len(n_grams) + 1:
                    break
                else:
                    best_path_length = len(path)
        return best_path, best_path_marked_edges

    def _transform_path_in_word(self, path, is_first_path):
        if is_first_path:
            y = path[0]
        else:
            y = ""
        for node in path[1:]:
            y += node[-1:]
        return y