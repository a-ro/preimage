__author__ = 'amelie'

from itertools import product

import numpy

from preimage.exceptions.n_gram import InvalidNGramLengthError


class Alphabet:
    latin = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
             'v', 'w', 'x', 'y', 'z']


def get_n_gram_to_index(alphabet, n):
    n_grams = get_n_grams(alphabet, n)
    indexes = numpy.arange(len(n_grams))
    n_gram_to_index = dict(zip(n_grams, indexes))
    return n_gram_to_index


def get_index_to_n_gram(alphabet, n):
    n_grams = get_n_grams(alphabet, n)
    indexes = numpy.arange(len(n_grams))
    index_to_n_gram = dict(zip(indexes, n_grams))
    return index_to_n_gram


def get_n_grams(alphabet, n):
    n = int(n)
    if n <= 0:
        raise InvalidNGramLengthError(n)
    n_grams = [''.join(n_gram) for n_gram in product(alphabet, repeat=n)]
    return numpy.array(n_grams)


def transform_strings_to_integer_lists(Y):
    n_examples = numpy.array(Y).shape[0]
    max_length = numpy.max([len(y) for y in Y])
    Y_int = numpy.zeros((n_examples, max_length), dtype=numpy.int8) - 1
    for y_index, y in enumerate(Y):
        for letter_index, letter in enumerate(y):
            Y_int[y_index, letter_index] = ord(letter)
    return Y_int