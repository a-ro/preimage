import numpy


cdef class Node:
    def __init__(self, y, bound, real_min_bound, real_max_bound):
        self.y = y
        self.bound = bound
        self.real_min_bound = real_min_bound
        self.real_max_bound = real_max_bound

    def __str__(self):
        node_string = "y: {}, bound: {}, real_min_bound: {}, real_max_bound: {}"
        return node_string.format(self.y, self.get_bound(), self.real_min_bound, self.real_max_bound)

    cdef FLOAT64_t get_bound(self):
        return self.bound


cdef class MaxNode(Node):
    def __init__(self, y, bound, real_min_bound, real_max_bound):
        Node.__init__(self, y, -bound, real_min_bound, real_max_bound)

    def __richcmp__(self, MaxNode other_node, int op):
        if op == 0:
            if self.bound == other_node.bound:
                return numpy.random.randint(2)
            else:
                return self.bound < other_node.bound
        if op == 2:
            return self.y == other_node.y and self.bound == other_node.bound \
                   and self.real_min_bound == other_node.real_min_bound \
                   and self.real_max_bound == other_node.real_max_bound

    cdef FLOAT64_t get_bound(self):
        return -self.bound

    cdef MinNode convert_to_min_node(self):
        return MinNode(self.y, self.get_bound(), self.real_min_bound, self.real_max_bound)


cdef class MinNode(Node):
    def __init__(self, y, bound, real_min_bound, real_max_bound):
        Node.__init__(self, y, bound, real_min_bound, real_max_bound)

    def __richcmp__(self, MinNode other_node, int op):
        if op == 0:
            if self.bound == other_node.bound:
                return numpy.random.randint(2)
            else:
                return self.bound < other_node.bound
        if op == 2:
            return self.y == other_node.y and self.bound == other_node.bound \
                   and self.real_min_bound == other_node.real_min_bound \
                   and self.real_max_bound == other_node.real_max_bound

    cdef MaxNode convert_to_max_node(self):
        return MaxNode(self.y, self.get_bound(), self.real_min_bound, self.real_max_bound)