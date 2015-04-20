__author__ = 'amelie'

import numpy

from preimage.inference.string_feature_space import build_feature_space_without_positions


# Sparse matrix representation of the n_grams in each word (y)
class NGramFeatureSpace:
    def __init__(self, alphabet, n, Y):
        self._Feature_space = build_feature_space_without_positions(alphabet, n, Y)

    def compute_weights(self, y_weights):
        data_copy = numpy.copy(self._Feature_space.data)
        self._Feature_space.data *= self._repeat_each_y_weight_by_y_column_count(y_weights)
        n_gram_weights = numpy.array(self._Feature_space.sum(axis=0))[0]
        self._Feature_space.data = data_copy
        return n_gram_weights

    def _repeat_each_y_weight_by_y_column_count(self, y_weights):
        return y_weights.repeat(numpy.diff(self._Feature_space.indptr))