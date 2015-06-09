cimport numpy


ctypedef numpy.float64_t FLOAT64_t


cdef class Node:
    cdef public str y
    cdef public FLOAT64_t real_min_bound, real_max_bound, bound
    cdef FLOAT64_t get_bound(self)


cdef class MaxNode(Node):
    cdef FLOAT64_t get_bound(self)
    cdef MinNode convert_to_min_node(self)


cdef class MinNode(Node):
    cdef MaxNode convert_to_max_node(self)