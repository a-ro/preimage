__author__ = 'amelie'

import numpy

from preimage.features.string_feature_space import build_feature_space_with_positions
from preimage.utils.position import compute_position_weights
from preimage.kernels.generic_string import element_wise_kernel


# Sparse matrix representation of the n_grams in each word (y). Takes in account the position of the n_grams
# Gives a matrix of shape m x max_n_gram_count_in_y*(len(alphabet)**n)
# Doesn't take in account the similarity between the n-grams (no sigma_c)
class GenericStringFeatureSpace:
    def __init__(self, alphabet, n, Y, sigma_position, is_normalized):
        self._n = int(n)
        self._sigma_position = sigma_position
        self._alphabet_n_gram_count = len(alphabet) ** n
        self._feature_space = build_feature_space_with_positions(alphabet, self._n, Y)
        self._max_n_gram_count = self._get_max_n_gram_count(self._alphabet_n_gram_count, self._feature_space)
        self._normalize(self._feature_space, self._n, Y, sigma_position, is_normalized)

    def _get_max_n_gram_count(self, alphabet_n_gram_count, feature_space):
        n_columns = feature_space.shape[1]
        max_n_gram_count = int(n_columns / alphabet_n_gram_count)
        return max_n_gram_count

    def _normalize(self, feature_space, n, Y, sigma_position, is_normalized):
        if is_normalized:
            y_y_similarity = element_wise_kernel(Y, sigma_position, n, is_blended=False)
            y_normalization = 1. / numpy.sqrt(y_y_similarity)
            data_normalization = y_normalization.repeat(numpy.diff(feature_space.indptr))
            feature_space.data *= data_normalization

    def _get_n_gram_count_in_each_y(self, n, Y):
        y_n_gram_counts = numpy.array([len(y) - n + 1 for y in Y])
        return y_n_gram_counts

    def compute_weights(self, y_weights, y_length):
        y_n_gram_count = y_length - self._n + 1
        data_copy = numpy.copy(self._feature_space.data)
        self._feature_space.data *= self._repeat_each_y_weight_by_y_column_count(y_weights)
        weighted_degree_weights = numpy.array(self._feature_space.sum(axis=0))[0].reshape(self._max_n_gram_count, -1)
        self._feature_space.data = data_copy
        gs_weights = self._transform_in_gs_weights(y_n_gram_count, weighted_degree_weights)
        return gs_weights

    def _transform_in_gs_weights(self, y_n_gram_count, weighted_degree_weights):
        gs_weights = numpy.empty((y_n_gram_count, self._alphabet_n_gram_count))
        for i in range(y_n_gram_count):
            position_weights = compute_position_weights(i, self._max_n_gram_count, self._sigma_position).reshape(-1, 1)
            gs_weights[i, :] = (weighted_degree_weights * position_weights).sum(axis=0)
        return gs_weights

    def _repeat_each_y_weight_by_y_column_count(self, y_weights):
        return y_weights.repeat(numpy.diff(self._feature_space.indptr))