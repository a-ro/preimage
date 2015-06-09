import numpy


cdef class MaxNode:
    # negative bound because we want to maximize
    def __cinit__(self, y, bound, real_min_bound, real_max_bound):
        self.y = y
        self.bound = -bound
        self.real_min_bound = real_min_bound
        self.real_max_bound = real_max_bound

    # use random when equal bounds (n-gram kernel might have a non-unique pre-image)
    # only < operators is implemented for heap sort and == operator for unit tests
    def __richcmp__(self, MaxNode other_node, int op):
        if op == 0:
            if self.bound == other_node.bound:
                return numpy.random.randint(2)
            else:
                return self.bound < other_node.bound
        elif op == 2:
            return self.y == other_node.y and self.bound == other_node.bound \
                   and self.real_min_bound == other_node.real_min_bound \
                   and self.real_max_bound == other_node.real_max_bound

    def __str__(self):
        node_string = "y: {}, bound: {}, real_min_bound: {}, real_max_bound: {}"
        return node_string.format(self.y, self.get_bound(), self.real_min_bound, self.real_max_bound)

    cdef MaxNode copy(self):
        return MaxNode(self.y, self.get_bound(), self.real_min_bound, self.real_max_bound)

    cdef FLOAT64_t get_bound(self):
        return -self.bound