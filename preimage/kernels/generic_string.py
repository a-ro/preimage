#!/usr/bin/env python
# -*- coding:utf-8 -*-

# GS kernel modified for string instead of amino acids (no n_gram similarity -> sigma_c = 0, not blended only n_gram)
# Original GS kernel Code was made by Sebastien Giguère

import numpy

from preimage.kernels._generic_string import element_wise_generic_string_kernel
from preimage.utils.position import compute_position_weights_matrix
from preimage.utils.alphabet import transform_strings_to_integer_lists


def element_wise_kernel(X, sigma_position, n):
    X = numpy.array(X)
    x_lengths = numpy.array([len(x) for x in X], dtype=numpy.int64)
    max_length = numpy.max(x_lengths) - n + 1
    position_matrix = compute_position_weights_matrix(max_length, sigma_position)
    X_int = transform_strings_to_integer_lists(X)
    kernel = element_wise_generic_string_kernel(X_int, x_lengths, position_matrix, n)
    return kernel