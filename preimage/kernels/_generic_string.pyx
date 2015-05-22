import cython
import numpy
cimport numpy

numpy.import_array()

ctypedef numpy.float64_t FLOAT64_t
ctypedef numpy.int8_t INT8_t
ctypedef numpy.int64_t INT64_t


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline FLOAT64_t generic_string_non_blended_similarity(INT8_t[::1] x1, INT64_t x1_length, INT8_t[::1] x2,
                                                            INT64_t x2_length, FLOAT64_t[:,::1] position_matrix,
                                                            INT64_t n):
    cdef int i,j,l, is_n_gram_equal
    cdef FLOAT64_t similarity = 0.
    for i in range(x1_length - n + 1):
        for j in range(x2_length - n + 1):
            is_n_gram_equal = 1
            for l in range(n):
                is_n_gram_equal *= x1[i + l] == x2[j + l]
            similarity += position_matrix[i, j] * is_n_gram_equal
    return similarity


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef element_wise_generic_string_kernel(INT8_t[:, ::1] X, INT64_t[::1] x_lengths, FLOAT64_t[:, ::1] position_matrix,
                                         INT64_t n):
    cdef int i
    cdef FLOAT64_t[::1] kernel = numpy.empty(X.shape[0], dtype=numpy.float64)
    for i in range(X.shape[0]):
        kernel[i] = generic_string_non_blended_similarity(X[i], x_lengths[i], X[i], x_lengths[i], position_matrix, n)
    return numpy.asarray(kernel)