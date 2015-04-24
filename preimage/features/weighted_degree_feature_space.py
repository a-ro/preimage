__author__ = 'amelie'

import numpy

from preimage.features.string_feature_space import build_feature_space_with_positions


# Sparse matrix representation of the n_grams in each word (y). Takes in account the position of the n_grams
# Gives a matrix of shape m x max_n_gram_count_in_y*(len(alphabet)**n)
class WeightedDegreeFeatureSpace:
    def __init__(self, alphabet, n, Y, is_normalized):
        self._n = int(n)
        self._alphabet_n_gram_count = len(alphabet) ** n
        self._is_normalized = is_normalized
        self._Feature_space = build_feature_space_with_positions(alphabet, self._n, Y)
        self._normalize(is_normalized, self._Feature_space)
        self._max_n_gram_count = self.__get_max_n_gram_count(self._alphabet_n_gram_count, self._Feature_space)

    def __get_max_n_gram_count(self, alphabet_n_gram_count, Feature_space):
        n_columns = Feature_space.shape[1]
        max_n_gram_count = int(n_columns / alphabet_n_gram_count)
        return max_n_gram_count

    def _normalize(self, is_normalized, Y_feature_space):
        if is_normalized:
            y_normalization = 1. / numpy.sqrt(numpy.array(Y_feature_space.sum(axis=1).reshape(1, -1))[0])
            data_normalization = y_normalization.repeat(numpy.diff(Y_feature_space.indptr))
            Y_feature_space.data *= data_normalization

    def compute_weights(self, y_weights, y_length):
        y_n_gram_count = y_length - self._n + 1
        data_copy = numpy.copy(self._Feature_space.data)
        self._Feature_space.data *= self._repeat_each_y_weight_by_y_column_count(y_weights)
        Weighted_degree_weights = numpy.array(self._Feature_space.sum(axis=0))[0]
        self._Feature_space.data = data_copy
        Weighted_degree_weights = self._get_weight_for_each_graph_partition(y_n_gram_count, Weighted_degree_weights)
        return Weighted_degree_weights

    def _get_weight_for_each_graph_partition(self, y_n_gram_count, Weighted_degree_weights):
        Weighted_degree_weights = Weighted_degree_weights.reshape(self._max_n_gram_count, -1)
        if y_n_gram_count <= self._max_n_gram_count:
            Weighted_degree_weights = Weighted_degree_weights[0:y_n_gram_count, :]
        else:
            Partitions_with_zero_weight = numpy.zeros((y_n_gram_count - self._max_n_gram_count,
                                                       self._alphabet_n_gram_count))
            Weighted_degree_weights = numpy.concatenate((Weighted_degree_weights, Partitions_with_zero_weight), axis=0)
        return Weighted_degree_weights

    def _repeat_each_y_weight_by_y_column_count(self, y_weights):
        return y_weights.repeat(numpy.diff(self._Feature_space.indptr))