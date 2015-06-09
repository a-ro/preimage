import cython
import numpy
cimport numpy
numpy.import_array()

ctypedef numpy.float64_t FLOAT64_t
ctypedef numpy.int8_t INT8_t
ctypedef numpy.int64_t INT64_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef FLOAT64_t[:,::1] compute_gs_similarity_weights(int n_partitions, INT8_t[:,::1] n_grams, INT8_t[:,::1] Y,
                                                     FLOAT64_t[::1] y_weights, INT64_t[::1] y_lengths,
                                                     FLOAT64_t[:,::1] position_matrix,
                                                     FLOAT64_t[:,::1] similarity_matrix):
    cdef int partition_index, n_gram_index
    cdef INT8_t[::1] n_gram
    cdef FLOAT64_t[:,::1] gs_weights = numpy.empty((n_partitions, n_grams.shape[0]))

    for partition_index in range(n_partitions):
        for n_gram_index, n_gram in enumerate(n_grams):
            if partition_index < n_partitions - 1:
                gs_weights[partition_index, n_gram_index] = compute_n_gram_weight(partition_index, n_gram, Y, y_weights,
                                                                                  y_lengths, position_matrix,
                                                                                  similarity_matrix)
            else:
                gs_weights[partition_index, n_gram_index] = compute_last_n_gram_weight(partition_index, n_gram, Y,
                                                                                       y_weights, y_lengths,
                                                                                       position_matrix,
                                                                                       similarity_matrix)
    return gs_weights


@cython.boundscheck(False)
@cython.wraparound(False)
cdef FLOAT64_t compute_n_gram_weight(int partition_index, INT8_t[::1] n_gram, INT8_t[:,::1] Y, FLOAT64_t[::1] y_weights,
                                     INT64_t[::1] y_lengths, FLOAT64_t[:,::1] position_matrix,
                                     FLOAT64_t[:,::1] similarity_matrix):
    cdef INT8_t[::1] y
    cdef FLOAT64_t kernel, n_gram_similarity, current_similarity, n_gram_weight
    cdef INT64_t y_length, max_length, n_gram_length
    cdef int y_index, i, j

    n_gram_weight = 0.
    n_gram_length = n_gram.shape[0]
    for y_index, y in enumerate(Y):
        kernel = 0.
        y_length = y_lengths[y_index]
        for i in range(y_length):
            max_length = min(n_gram_length, y_length - i)
            current_similarity = 1.
            n_gram_similarity = 0.
            for j in range(max_length):
                current_similarity *=  similarity_matrix[n_gram[j], y[i + j]]
                n_gram_similarity += current_similarity
            kernel += position_matrix[partition_index, i] * n_gram_similarity
        n_gram_weight += y_weights[y_index] * kernel
    return n_gram_weight


@cython.boundscheck(False)
@cython.wraparound(False)
cdef FLOAT64_t compute_last_n_gram_weight(int partition_index, INT8_t[::1] n_gram, INT8_t[:,::1] Y,
                                          FLOAT64_t[::1] y_weights, INT64_t[::1] y_lengths,
                                          FLOAT64_t[:,::1] position_matrix, FLOAT64_t[:,::1] similarity_matrix):
    cdef FLOAT64_t n_gram_weight
    cdef int i
    n_gram_weight = 0.0
    for i in range(n_gram.shape[0]):
        n_gram_weight += compute_n_gram_weight(i + partition_index, n_gram[i:], Y, y_weights, y_lengths,
                                               position_matrix, similarity_matrix)
    return n_gram_weight