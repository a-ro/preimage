__author__ = 'amelie'

import numpy
from preimage.features.gs_similarity_weights import compute_gs_similarity_weights
from preimage.utils.alphabet import transform_strings_to_integer_lists, get_n_grams
# Here we can't use a sparse matrix representation because we take in account the similarity between the n-grams
# so there is no non-zero n-grams like in the other output feature spaces.

class GenericStringSimilarityFeatureSpace:
    def __init__(self, alphabet, n, Y, is_normalized, gs_kernel):
        self._n = int(n)
        self._is_normalized = is_normalized
        self._Y_int = transform_strings_to_integer_lists(Y, alphabet)
        self._n_grams_int = transform_strings_to_integer_lists(get_n_grams(alphabet, n), alphabet)
        self._y_lengths = numpy.array([len(y) for y in Y])
        self._max_train_length = numpy.max(self._y_lengths)
        self._n_gram_similarity_matrix = gs_kernel.get_alphabet_similarity_matrix()
        self._gs_kernel = gs_kernel
        if is_normalized:
            self._normalization = numpy.sqrt(gs_kernel.element_wise_kernel(Y))

    def compute_weights(self, y_weights, y_length):
        normalized_weights = numpy.copy(y_weights)
        max_length = max(y_length, self._max_train_length)
        if self._is_normalized:
            normalized_weights *= 1. / self._normalization
        n_partitions = y_length - self._n + 1
        position_matrix = self._gs_kernel.get_position_matrix(max_length)
        gs_weights = compute_gs_similarity_weights(n_partitions, self._n_grams_int, self._Y_int, normalized_weights,
                                                   self._y_lengths, position_matrix, self._n_gram_similarity_matrix)
        return numpy.array(gs_weights)