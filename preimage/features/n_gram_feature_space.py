__author__ = 'amelie'

import numpy

from preimage.features.string_feature_space import build_feature_space_without_positions


# Sparse matrix representation of the n_grams in each word (y)
class NGramFeatureSpace:
    def __init__(self, alphabet, n, Y, is_normalized):
        self._Feature_space = build_feature_space_without_positions(alphabet, n, Y)
        self._normalize(is_normalized, self._Feature_space)

    def _normalize(self, is_normalized, Y_feature_space):
        if is_normalized:
            y_normalization = self._get_y_normalization(Y_feature_space)
            data_normalization = y_normalization.repeat(numpy.diff(Y_feature_space.indptr))
            Y_feature_space.data *= data_normalization

    def _get_y_normalization(self, Y_feature_space):
        y_normalization = (Y_feature_space.multiply(Y_feature_space)).sum(axis=1)
        y_normalization = 1. / numpy.sqrt(numpy.array((y_normalization.reshape(1, -1))[0]))
        return y_normalization

    def compute_weights(self, y_weights):
        data_copy = numpy.copy(self._Feature_space.data)
        self._Feature_space.data *= self._repeat_each_y_weight_by_y_column_count(y_weights)
        n_gram_weights = numpy.array(self._Feature_space.sum(axis=0))[0]
        self._Feature_space.data = data_copy
        return n_gram_weights

    def _repeat_each_y_weight_by_y_column_count(self, y_weights):
        return y_weights.repeat(numpy.diff(self._Feature_space.indptr))