cimport numpy
from _node cimport MaxNode
from _bound_calculator cimport BoundCalculator, Bound, FLOAT64_t


cdef class NodeCreator:
    cdef:
        BoundCalculator min_bound_calculator
        BoundCalculator max_bound_calculator
        list n_grams

    cdef MaxNode create_node(self, str  y, MaxNode parent_node, int final_length)

    cdef list get_start_nodes(self, int final_length)