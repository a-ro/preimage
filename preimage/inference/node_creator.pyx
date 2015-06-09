from libc.math cimport sqrt


cdef class NodeCreator:
    def __init__(self, min_bound_calculator, max_bound_calculator, n_grams):
        self.min_bound_calculator = min_bound_calculator
        self.max_bound_calculator = max_bound_calculator
        self.n_grams = n_grams

    # For unit tests only
    def create_node_python(self, str  y, MaxNode parent_node, int final_length):
        return self.create_node(y, parent_node, final_length)

    cdef MaxNode create_node(self, str  y, MaxNode parent_node, int final_length):
        cdef Bound max_bound = self.max_bound_calculator.compute_bound(y, parent_node.real_max_bound, final_length)
        cdef Bound min_bound = self.min_bound_calculator.compute_bound(y, parent_node.real_min_bound, final_length)
        cdef FLOAT64_t bound_value = max_bound.bound_value / sqrt(min_bound.bound_value)
        return MaxNode(y, bound_value, min_bound.real_value, max_bound.real_value)

    # For unit tests only
    def get_start_nodes_python(self, int final_length):
        return self.get_start_nodes(final_length)

    cdef list get_start_nodes(self, int final_length):
        cdef list start_nodes = []
        cdef FLOAT64_t[::1] max_values = self.max_bound_calculator.get_start_node_real_values(final_length)
        cdef FLOAT64_t[::1] max_bounds = self.max_bound_calculator.get_start_node_bounds(final_length)
        cdef FLOAT64_t[::1] min_values = self.min_bound_calculator.get_start_node_real_values(final_length)
        cdef FLOAT64_t[::1] min_bounds = self.min_bound_calculator.get_start_node_bounds(final_length)
        for i in range(len(self.n_grams)):
            start_nodes.append(MaxNode(str(self.n_grams[i]), max_bounds[i] / sqrt(min_bounds[i]), min_values[i],
                                       max_values[i]))
        return start_nodes


# For unit tests only
cdef class NodeCreatorMock(NodeCreator):
    cdef mock

    def __init__(self, mock):
        self.mock = mock

    cdef MaxNode create_node(self, str  y, MaxNode parent_node, int final_length):
        return self.mock.create_node(y, final_length)

    cdef list get_start_nodes(self, int final_length):
        return self.mock.get_start_nodes(final_length)