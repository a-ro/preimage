__author__ = 'amelie'

import numpy

from preimage.features.string_feature_space import build_feature_space_with_positions
from preimage.utils.position import compute_position_weights


# Sparse matrix representation of the n_grams in each word (y). Takes in account the position of the n_grams
# Gives a matrix of shape m x max_n_gram_count_in_y*(len(alphabet)**n)
class GenericStringFeatureSpace:
    def __init__(self, alphabet, n, Y, sigma_position):
        self._n = int(n)
        self.sigma_position = sigma_position
        self._alphabet_n_gram_count = len(alphabet) ** n
        self._Feature_space = build_feature_space_with_positions(alphabet, self._n, Y)
        self._max_n_gram_count = self.__get_max_n_gram_count(self._alphabet_n_gram_count, self._Feature_space)

    def __get_max_n_gram_count(self, alphabet_n_gram_count, Feature_space):
        n_columns = Feature_space.shape[1]
        max_n_gram_count = int(n_columns / alphabet_n_gram_count)
        return max_n_gram_count

    def compute_weights(self, y_weights, y_length):
        y_n_gram_count = y_length - self._n + 1
        data_copy = numpy.copy(self._Feature_space.data)
        self._Feature_space.data *= self._repeat_each_y_weight_by_y_column_count(y_weights)
        Weighted_degree_weights = numpy.array(self._Feature_space.sum(axis=0))[0].reshape(self._max_n_gram_count, -1)
        self._Feature_space.data = data_copy
        GS_weights = self._transform_in_gs_weights(y_n_gram_count, Weighted_degree_weights)
        return GS_weights

    def _transform_in_gs_weights(self, y_n_gram_count, Weighted_degree_weights):
        GS_weights = numpy.empty((y_n_gram_count, self._alphabet_n_gram_count))
        for i in range(y_n_gram_count):
            position_weights = compute_position_weights(i, self._max_n_gram_count, self.sigma_position).reshape(-1, 1)
            GS_weights[i, :] = (Weighted_degree_weights * position_weights).sum(axis=0)
        return GS_weights

    def _repeat_each_y_weight_by_y_column_count(self, y_weights):
        return y_weights.repeat(numpy.diff(self._Feature_space.indptr))