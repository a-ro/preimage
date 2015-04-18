__author__ = 'amelie'

import numpy
from scipy.sparse import csr_matrix

from preimage.utils.alphabet import get_n_gram_to_index
from preimage.exceptions.n_gram import InvalidNGramError


# Sparse matrix representation of the n_grams in each word (y). Takes in account the position of the n_grams
# Gives a matrix of shape m x max_n_gram_count_in_y*(len(alphabet)**n)
# todo add compute weights
# todo combine n_gram and weighted feature space to avoid repetitive code
class WeightedDegreeFeatureSpace:
    def __init__(self, n, alphabet, Y):
        n = int(n)
        total_n_gram_count = len(alphabet)**n
        n_gram_to_index = get_n_gram_to_index(alphabet, n)
        self._Feature_space = self._build_feature_space(n, n_gram_to_index, numpy.array(Y), total_n_gram_count)

    def _build_feature_space(self, n, n_gram_to_index, Y, total_n_gram_count):
        n_examples = Y.shape[0]
        indice_pointers, columns, data = self._initialize_indice_pointers_columns_data(n_examples)
        max_n_gram_in_word = self._get_max_number_of_n_gram_in_words(Y, n)
        for y_index, y in enumerate(Y):
            self._build_y_indice_pointer_columns_data(indice_pointers, columns, data, n, n_gram_to_index, y, y_index,
                                                      total_n_gram_count)
        N_gram_feature_space = csr_matrix((numpy.array(data), numpy.array(columns), numpy.array(indice_pointers)),
                                          shape=(n_examples, max_n_gram_in_word*total_n_gram_count), dtype=numpy.float)
        return N_gram_feature_space

    def _get_max_number_of_n_gram_in_words(self, Y, n):
        max_word_length = numpy.max([len(y) for y in Y])
        max_n_gram_count = max_word_length - n + 1
        return max_n_gram_count

    def _initialize_indice_pointers_columns_data(self, n_examples):
        indice_pointers = numpy.empty(n_examples + 1, dtype=numpy.int)
        indice_pointers[0] = 0
        columns = []
        data = []
        return indice_pointers, columns, data

    def _build_y_indice_pointer_columns_data(self, indice_pointers, columns, data, n, n_gram_to_index, y, y_index,
                                             total_n_gram_count):
        y_n_gram_count = len(y) - n + 1
        indice_pointers[y_index + 1] = indice_pointers[y_index] + y_n_gram_count
        columns += self._get_y_column_indexes(y, n, n_gram_to_index, total_n_gram_count, y_n_gram_count)
        data += [1.] * y_n_gram_count

    def _get_y_column_indexes(self, y, n, n_gram_to_index, total_n_gram_count, y_n_gram_count):
        try:
            column_indexes = [(i*total_n_gram_count) + n_gram_to_index[y[i:i + n]] for i in range(y_n_gram_count)]
        except KeyError as key_error:
            raise InvalidNGramError(n, key_error.args[0])
        return column_indexes