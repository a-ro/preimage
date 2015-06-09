from libc.math cimport sqrt
from node cimport MaxNode
cimport numpy
import numpy


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
        graph_weight_partition_index = min(self.graph_weights.shape[0]-1, max_partition_index)
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
        for i in range(1, len(y)-self.n+1):
            if y[0:self.n] == y[i:i+self.n]:
                similarity += self.position_weights[i]
        return 1 + 2 * similarity

    cdef FLOAT64_t[::1] get_start_node_real_values(self, int final_length):
        return numpy.ones(len(self.n_grams))

    cdef FLOAT64_t[::1] get_start_node_bounds(self, int final_length):
        return numpy.ones(len(self.n_grams)) * (final_length - self.n + 1)


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