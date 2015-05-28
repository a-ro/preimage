cimport numpy


ctypedef numpy.float64_t FLOAT64_t


cdef class MaxNode:
    cdef public str y
    cdef public FLOAT64_t real_min_bound, real_max_bound, bound

    cdef MaxNode copy(self)

    cdef FLOAT64_t get_bound(self)