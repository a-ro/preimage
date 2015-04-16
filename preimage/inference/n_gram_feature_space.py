__author__ = 'amelie'

from collections import Counter

import numpy
from scipy.sparse import csr_matrix

from preimage.utils.alphabet import get_n_gram_to_index
from preimage.exceptions.n_gram import InvalidNGramError


# Sparse matrix representation of the n_grams in each word (y)
class NGramFeatureSpace:
    def __init__(self, n, alphabet, Y):
        n = int(n)
        n_gram_to_index = get_n_gram_to_index(alphabet, n)
        self._N_gram_feature_space = self._build_feature_space(n, n_gram_to_index, numpy.array(Y))

    def _build_feature_space(self, n, n_gram_to_index, Y):
        n_examples = Y.shape[0]
        indice_pointers, columns, data = self._initialize_indice_pointers_columns_data(n_examples)
        for y_index, y in enumerate(Y):
            self._build_y_indice_pointer_columns_data(indice_pointers, columns, data, n, n_gram_to_index, y, y_index)
        N_gram_feature_space = csr_matrix((numpy.array(data), numpy.array(columns), numpy.array(indice_pointers)),
                                          shape=(n_examples, len(n_gram_to_index)))
        return N_gram_feature_space

    def _initialize_indice_pointers_columns_data(self, n_examples):
        indice_pointers = numpy.empty(n_examples + 1, dtype=numpy.int)
        indice_pointers[0] = 0
        columns = []
        data = []
        return indice_pointers, columns, data

    def _build_y_indice_pointer_columns_data(self, indice_pointers, columns, data, n, n_gram_to_index, y, y_index):
        y_column_indexes = self._get_y_column_indexes(y, n, n_gram_to_index)
        y_unique_column_indexes = list(numpy.unique(y_column_indexes))
        indice_pointers[y_index + 1] = self._get_y_indice_pointer(y_index, indice_pointers, y_unique_column_indexes)
        columns += y_unique_column_indexes
        data += self._get_y_data(y_column_indexes, y_unique_column_indexes)

    def _get_y_indice_pointer(self, y_index, indice_pointers, y_unique_column_indexes):
        indice_pointer = indice_pointers[y_index] + len(y_unique_column_indexes)
        return indice_pointer

    def _get_y_column_indexes(self, y, n, n_gram_to_index):
        y_n_gram_count = len(y) - n + 1
        try:
            column_indexes = [n_gram_to_index[y[i:i + n]] for i in range(y_n_gram_count)]
        except KeyError as key_error:
            raise InvalidNGramError(key_error.args[0], n)
        return column_indexes

    # There is probably a faster way to do this but it must keep the index_count in index increasing order
    def _get_y_data(self, y_n_gram_indexes, unique_n_gram_indexes):
        n_gram_counter = Counter(y_n_gram_indexes)
        data = [n_gram_counter[index] for index in unique_n_gram_indexes]
        return data