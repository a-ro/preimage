cimport numpy
from _node cimport MaxNode


ctypedef numpy.float64_t FLOAT64_t


cdef struct Bound:
    FLOAT64_t bound_value
    FLOAT64_t real_value


cdef class BoundCalculator:
    cdef Bound compute_bound(self, str  y, FLOAT64_t parent_real_value, int final_length)

    cdef FLOAT64_t[::1] get_start_node_real_values(self, int final_length)

    cdef FLOAT64_t[::1] get_start_node_bounds(self, int final_length)


cdef class MaxBoundCalculator(BoundCalculator):
    cdef:
        FLOAT64_t[:,::1] graph, graph_weights
        int n
        dict n_gram_to_index


cdef class OCRMinBoundCalculator(BoundCalculator):
    cdef:
        FLOAT64_t[::1] position_weights
        int n
        list n_grams

    cdef FLOAT64_t gs_similarity_new_n_gram(self, str y)