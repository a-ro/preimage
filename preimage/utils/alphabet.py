__author__ = 'amelie'

from itertools import product

import numpy


class Alphabet:
    latin = numpy.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                         'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])


def get_n_gram_to_index(alphabet, n):
    n_grams = get_n_grams(alphabet, n)
    indexes = numpy.arange(len(n_grams))
    n_gram_to_index = dict(zip(n_grams, indexes))
    return n_gram_to_index


def get_n_grams(alphabet, n):
    n = int(n)
    if n <= 0:
        raise ValueError('n must be greater than zero. Got: n={:d}'.format(n))
    n_grams = [''.join(n_gram) for n_gram in product(alphabet, repeat=n)]
    return numpy.array(n_grams)