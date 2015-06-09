from libc.math cimport sqrt

cimport numpy
import numpy

from node cimport MaxNode


cdef class BoundCalculator:
    cdef Bound compute_bound(self, str  y, FLOAT64_t parent_real_value, int final_length):
         raise NotImplementedError('Subclasses should implement this method')

    cdef FLOAT64_t[::1] get_start_node_real_values(self, int final_length):
        raise NotImplementedError('Subclasses should implement this method')

    cdef FLOAT64_t[::1] get_start_node_bounds(self, int final_length):
        raise NotImplementedError('Subclasses should implement this method')

    # For unit tests only
    def compute_bound_python(self, str  y, FLOAT64_t parent_real_value, int final_length):
        return self.compute_bound(y, parent_real_value, final_length)

    # For unit tests only
    def get_start_node_real_values_python(self, int final_length):
        return numpy.array(self.get_start_node_real_values(final_length))

    # For unit tests only
    def get_start_node_bounds_python(self, int final_length):
        return numpy.array(self.get_start_node_bounds(final_length))


cdef class MaxBoundCalculator(BoundCalculator):
    def __init__(self, n, graph, graph_weights, n_gram_to_index):
        self.n = n
        self.graph = graph
        self.graph_weights = graph_weights
        self.n_gram_to_index = n_gram_to_index

    cdef Bound compute_bound(self, str  y, FLOAT64_t parent_real_value, int final_length):
        cdef int max_partition_index = final_length - self.n
        cdef int partition_index = max_partition_index - (len(y) - self.n)
        cdef int graph_weight_partition_index = min(self.graph_weights.shape[0]-1, partition_index)
        cdef int n_gram_index = self.n_gram_to_index[y[0:self.n]]
        cdef Bound max_bound
        max_bound.real_value = self.graph_weights[graph_weight_partition_index, n_gram_index] + parent_real_value
        max_bound.bound_value = self.graph[partition_index, n_gram_index] + parent_real_value
        return max_bound

    cdef FLOAT64_t[::1] get_start_node_real_values(self, int final_length):
        cdef int max_partition_index, graph_weight_partition_index
        max_partition_index = final_length - self.n
        graph_weight_partition_index = min(self.graph_weights.shape[0] - 1, max_partition_index)
        return self.graph_weights[graph_weight_partition_index]

    cdef FLOAT64_t[::1] get_start_node_bounds(self, int final_length):
        return self.graph[final_length - self.n]


cdef class OCRMinBoundCalculator(BoundCalculator):
    def __init__(self, n, position_weights, n_grams):
        self.n = n
        self.position_weights = position_weights
        self.n_grams = n_grams

    # In our experiments, we consider that YY' is zero since |A| > |y|
    cdef Bound compute_bound(self, str y, FLOAT64_t parent_real_value, int final_length):
        cdef FLOAT64_t gs_similarity = parent_real_value + self.gs_similarity_new_n_gram(y)
        cdef FLOAT64_t y_y_similarity = final_length - len(y)
        cdef Bound bound
        bound.bound_value = y_y_similarity + gs_similarity
        bound.real_value = gs_similarity
        return bound

    # Similarity only for the n-gram comparison (not blended)
    cdef FLOAT64_t gs_similarity_new_n_gram(self, str y):
        cdef int i
        cdef FLOAT64_t similarity = 0.
        for i in range(1, len(y) - self.n + 1):
            if y[0:self.n] == y[i:i + self.n]:
                similarity += self.position_weights[i]
        return 1 + 2 * similarity

    cdef FLOAT64_t[::1] get_start_node_real_values(self, int final_length):
        return numpy.ones(len(self.n_grams))

    cdef FLOAT64_t[::1] get_start_node_bounds(self, int final_length):
        return numpy.ones(len(self.n_grams)) * (final_length - self.n + 1)


cdef class PeptideMinBoundCalculator(BoundCalculator):
    def __init__(self, n, alphabet_length, n_grams, letter_to_index, final_length, gs_kernel):
        self.n = n
        self.alphabet_length = alphabet_length
        self.letter_to_index = letter_to_index
        self.similarity_matrix = gs_kernel.get_alphabet_similarity_matrix()
        self.position_matrix = gs_kernel.get_position_matrix(final_length)
        self.start_node_real_values = gs_kernel.element_wise_kernel(n_grams)
        self.y_y_bounds = self.precompute_y_y_bound_for_each_length(final_length)
        self.start_node_bound_values = self.precompute_start_node_bounds(final_length, n_grams)

    cdef FLOAT64_t[::1] precompute_y_y_bound_for_each_length(self, int max_length):
        cdef FLOAT64_t[::1] y_y_bounds = numpy.zeros(max_length, dtype=numpy.float64)
        cdef FLOAT64_t min_similarity = numpy.min(self.similarity_matrix)
        cdef FLOAT64_t length_bound, current_similarity
        cdef int length, j, k, l
        for length in range(1, max_length):
            length_bound = 0
            for j in range(length):
                for k in range(j + 1, length):
                    current_similarity = 1.
                    for l in range(self.n):
                        current_similarity *= min_similarity
                        length_bound += self.position_matrix[j, k] * current_similarity
            y_y_bounds[length] =  self.n * length + 2 * length_bound
        return y_y_bounds

    cdef FLOAT64_t[::1] precompute_start_node_bounds(self, int final_length, list n_grams):
        cdef FLOAT64_t[::1] bounds = numpy.empty(len(n_grams), dtype=numpy.float64)
        cdef FLOAT64_t y_y_prime_bound
        cdef str n_gram
        cdef int i
        cdef int n_gram_start_index = final_length - self.n
        for i, n_gram in enumerate(n_grams):
            y_y_prime_bound = self.compute_y_y_prime_bound(n_gram, n_gram_start_index)
            bounds[i] = self.start_node_real_values[i] + self.y_y_bounds[n_gram_start_index] + 2 * y_y_prime_bound
        return bounds

    cdef Bound compute_bound(self, str y, FLOAT64_t parent_real_value, int final_length):
        cdef FLOAT64_t gs_similarity = parent_real_value + self.gs_similarity_new_n_gram(y)
        cdef FLOAT64_t y_y_similarity = self.y_y_bounds[final_length - len(y)]
        cdef FLOAT64_t y_y_prime_similarity = self.compute_y_y_prime_bound(y, final_length - len(y))
        cdef Bound bound
        bound.bound_value = gs_similarity + y_y_similarity + 2 * y_y_prime_similarity
        bound.real_value = gs_similarity
        return bound

    cdef FLOAT64_t gs_similarity_new_n_gram(self, str y):
        cdef int i, l, max_length, index_one, index_two
        cdef FLOAT64_t current_similarity, n_gram_similarity
        cdef FLOAT64_t similarity = 0.
        for i in range(1, len(y)):
            max_length = min(self.n, len(y) - i)
            current_similarity = 1.
            n_gram_similarity = 0.
            for l in range(max_length):
                index_one = self.letter_to_index[y[l]]
                index_two = self.letter_to_index[y[i + l]]
                current_similarity *= self.similarity_matrix[index_one, index_two]
                n_gram_similarity += current_similarity
            similarity += self.position_matrix[0, i] * n_gram_similarity
        return self.n + 2 * similarity

    cdef FLOAT64_t compute_y_y_prime_bound(self, str y, int y_start_index):
        cdef numpy.ndarray[FLOAT64_t, ndim=2] similarity_matrix = numpy.asarray(self.similarity_matrix)
        cdef int i, n_gram_length
        cdef FLOAT64_t y_y_prime_bound = 0
        for n_gram_length in range(1, self.n + 1):
            for i in range(y_start_index):
                y_y_prime_bound += self.compute_n_gram_y_y_prime_bound(n_gram_length, i, y, y_start_index,
                                                                       similarity_matrix)
        return y_y_prime_bound

    cdef FLOAT64_t compute_n_gram_y_y_prime_bound(self, int n_gram_length, int n_gram_index, str y, int y_start_index,
                                                  numpy.ndarray[FLOAT64_t, ndim=2] similarity_matrix):
        cdef int i, l, letter_index
        cdef str letter
        cdef numpy.ndarray[FLOAT64_t, ndim=1] n_gram_scores, final_scores
        final_scores = numpy.zeros(self.alphabet_length ** n_gram_length)
        for i in range(len(y) - n_gram_length + 1):
            n_gram_scores = numpy.ones(self.alphabet_length ** n_gram_length)
            for l in range(n_gram_length):
                letter_index = self.letter_to_index[y[i + l]]
                n_gram_scores *= self.transform_letter_scores_in_n_gram_scores(similarity_matrix[letter_index, :],
                                                                               n_gram_length, l)
            final_scores += self.position_matrix[n_gram_index, y_start_index + i] * n_gram_scores
        return numpy.min(final_scores)

    cdef numpy.ndarray[FLOAT64_t, ndim=1] transform_letter_scores_in_n_gram_scores(self, numpy.ndarray[FLOAT64_t, ndim=1]
                                                                                   letter_scores, int n_gram_length,
                                                                                   int index_in_n_gram):
        cdef numpy.ndarray[FLOAT64_t, ndim=1] n_gram_scores
        cdef int n_repeat, n_tile
        n_repeat = self.alphabet_length ** (n_gram_length - index_in_n_gram - 1)
        n_tile = self.alphabet_length ** index_in_n_gram
        n_gram_scores = numpy.tile(numpy.repeat(letter_scores, n_repeat), n_tile)
        return n_gram_scores

    cdef FLOAT64_t[::1] get_start_node_real_values(self, int final_length):
        return self.start_node_real_values

    cdef FLOAT64_t[::1] get_start_node_bounds(self, int final_length):
        return self.start_node_bound_values


# For unit tests only
cdef class BoundCalculatorMock(BoundCalculator):
    cdef mock

    def __init__(self, mock):
        self.mock = mock

    cdef Bound compute_bound(self, str y, FLOAT64_t parent_real_value, int final_length):
        return self.mock.compute_bound(y, final_length)

    cdef FLOAT64_t[::1] get_start_node_real_values(self, int final_length):
        return self.mock.get_start_node_real_values(final_length)

    cdef FLOAT64_t[::1] get_start_node_bounds(self, int final_length):
        return self.mock.get_start_node_bounds(final_length)