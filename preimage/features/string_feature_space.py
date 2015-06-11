"""Builder of string feature space represented as sparse matrix"""

__author__ = 'amelie'

from collections import Counter

import numpy
from scipy.sparse import csr_matrix

from preimage.utils.alphabet import get_n_gram_to_index
from preimage.exceptions.n_gram import InvalidNGramError


def build_feature_space_without_positions(alphabet, n, Y):
    """Create the feature space without considering the position of the n-gram in the strings

    Parameters
    ----------
    alphabet : list
        List of letters.
    n : int
        N-gram length.
    Y : array, [n_samples, ]
        Training strings.

    Returns
    -------
    feature_space : sparse matrix, shape = [n_samples, len(alphabet)**n]
        Sparse matrix representation of the n-grams in each training string, where n_samples is the number of samples
        in Y.
    """
    n = int(n)
    n_examples = numpy.array(Y).shape[0]
    n_gram_to_index = get_n_gram_to_index(alphabet, n)
    index_pointers, indexes, data = __initialize_pointers_indexes_data(n_examples)
    __build_y_pointers_indexes_data(index_pointers, indexes, data, n, n_gram_to_index, Y)
    feature_space = __build_csr_matrix(index_pointers, indexes, data, n_examples, len(n_gram_to_index))
    return feature_space


def __initialize_pointers_indexes_data(n_examples):
    index_pointers = numpy.empty(n_examples + 1, dtype=numpy.int)
    index_pointers[0] = 0
    indexes = []
    data = []
    return index_pointers, indexes, data


def __build_y_pointers_indexes_data(index_pointers, indexes, data, n, n_gram_to_index, Y):
    for y_index, y in enumerate(Y):
        y_column_indexes = __get_y_indexes(y, n, n_gram_to_index)
        y_unique_column_indexes = list(numpy.unique(y_column_indexes))
        index_pointers[y_index + 1] = __get_y_index_pointer(y_index, index_pointers, y_unique_column_indexes)
        indexes += y_unique_column_indexes
        data += __get_y_data(y_column_indexes, y_unique_column_indexes)


def __get_y_index_pointer(y_index, index_pointers, y_unique_column_indexes):
    index_pointer = index_pointers[y_index] + len(y_unique_column_indexes)
    return index_pointer


def __get_y_indexes(y, n, n_gram_to_index):
    y_n_gram_count = len(y) - n + 1
    try:
        column_indexes = [n_gram_to_index[y[i:i + n]] for i in range(y_n_gram_count)]
    except KeyError as key_error:
        raise InvalidNGramError(n, key_error.args[0])
    return column_indexes


def __get_y_data(y_n_gram_indexes, unique_n_gram_indexes):
    n_gram_counter = Counter(y_n_gram_indexes)
    data = [n_gram_counter[index] for index in unique_n_gram_indexes]
    return data


def __build_csr_matrix(index_pointers, indexes, data, n_rows, n_columns):
    return csr_matrix((numpy.array(data), numpy.array(indexes), numpy.array(index_pointers)),
                      shape=(n_rows, n_columns), dtype=numpy.float)


def build_feature_space_with_positions(alphabet, n, Y):
    """Create the feature space by considering the position of the n-gram in the strings

    Parameters
    ----------
    alphabet : list
        list of letters
    n : int
        n-gram length
    Y : array, [n_samples, ]
        The training strings.

    Returns
    -------
    feature_space : sparse matrix, shape = [n_samples, max_n_gram_count * len(alphabet)**n]
        Sparse matrix representation of the n-grams in each string of Y, where n_samples is the number of training
        samples and max_n_gram_count is the number of n-gram in the highest length string of Y.
    """
    n = int(n)
    n_examples = numpy.array(Y).shape[0]
    n_gram_to_index = get_n_gram_to_index(alphabet, n)
    index_pointers, indexes, data = __initialize_pointers_indexes_data(n_examples)
    __build_pointers_indexes_data_with_positions(index_pointers, indexes, data, n, n_gram_to_index, Y)
    n_columns = __get_n_columns(n, len(n_gram_to_index), Y)
    feature_space = __build_csr_matrix(index_pointers, indexes, data, n_examples, n_columns)
    return feature_space


def __get_n_columns(n, n_gram_count_in_alphabet, Y):
    y_max_n_gram_count = numpy.max([len(y) for y in Y]) - n + 1
    n_columns = y_max_n_gram_count * n_gram_count_in_alphabet
    return n_columns


def __build_pointers_indexes_data_with_positions(index_pointers, indexes, data, n, n_gram_to_index, Y):
    for y_index, y in enumerate(Y):
        y_n_gram_count = len(y) - n + 1
        index_pointers[y_index + 1] = index_pointers[y_index] + y_n_gram_count
        indexes += __get_y_column_indexes_with_positions(y, n, n_gram_to_index, y_n_gram_count)
        data += [1.] * y_n_gram_count


def __get_y_column_indexes_with_positions(y, n, n_gram_to_index, y_n_gram_count):
    try:
        column_indexes = [(i * len(n_gram_to_index)) + n_gram_to_index[y[i:i + n]] for i in range(y_n_gram_count)]
    except KeyError as key_error:
        raise InvalidNGramError(n, key_error.args[0])
    return column_indexes