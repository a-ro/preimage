import cython
import heapq
import time

import numpy
cimport numpy

from preimage.inference.node cimport MaxNode, MinNode
from preimage.inference.node_creator cimport NodeCreator

ctypedef numpy.float64_t FLOAT64_t


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef branch_and_bound(NodeCreator node_creator, int y_length, list alphabet, float max_time):
    cdef float start_time = time.clock()
    cdef float current_time = time.clock() - start_time
    cdef str empty_string = ""
    cdef MaxNode node, best_node
    cdef list heap = node_creator.get_start_nodes(y_length)

    heapq.heapify(heap)
    best_node = MaxNode(empty_string, -numpy.inf, 0, 0)
    while(len(heap) > 0 and current_time < max_time):
        node = heapq.heappop(heap)
        if  best_node < node:
            break
        node = depth_first_search(node, best_node, node_creator, heap, y_length, y_length, alphabet)
        if node < best_node and len(node.y) == y_length:
            best_node = node
        current_time = time.clock() - start_time
    return best_node.y, best_node.get_bound()


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef branch_and_bound_no_length(NodeCreator node_creator, int min_length, int max_length, list alphabet,
                                 float max_time):
    cdef float start_time = time.clock()
    cdef float current_time = time.clock() - start_time
    cdef str empty_string = ""
    cdef MaxNode best_node
    cdef int key, length
    cdef list heap
    cdef list keys_to_remove
    cdef dict heaps

    heaps = get_heap_for_each_length(node_creator, min_length, max_length)
    best_node = MaxNode(empty_string, -numpy.inf, 0, 0)
    while(len(heaps) > 0 and current_time < max_time):
        keys_to_remove = []
        for length, heap in heaps.items():
            best_node = find_length_best_node(best_node, node_creator, heap, length, alphabet, keys_to_remove)
            current_time = time.clock() - start_time
            if current_time > max_time:
                break
        for key in keys_to_remove:
            heaps.pop(key)
        current_time = time.clock() - start_time
    return best_node.y, best_node.get_bound()


@cython.boundscheck(False)
@cython.wraparound(False)
cdef dict get_heap_for_each_length(NodeCreator node_creator, int min_length, int max_length):
    cdef list heap
    cdef list keys_to_remove
    cdef dict heaps = {}
    cdef int length

    for length in range(min_length, max_length + 1):
        heap = node_creator.get_start_nodes(length)
        heapq.heapify(heap)
        heaps[length] = heap
    return heaps


@cython.boundscheck(False)
@cython.wraparound(False)
cdef MaxNode find_length_best_node(MaxNode best_node, NodeCreator node_creator, list heap, int length,
                                   list alphabet, list keys_to_remove):
    cdef MaxNode node

    if len(heap) == 0:
         keys_to_remove.append(length)
    else:
        node = heapq.heappop(heap)
        if best_node < node:
            keys_to_remove.append(length)
        else:
            node = depth_first_search(node, best_node, node_creator, heap, length, length, alphabet)
            if node < best_node and len(node.y) == length:
                best_node = node
    return best_node


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef branch_and_bound_multiple_solutions(NodeCreator node_creator, int y_length, int n_solutions, list alphabet,
                                          float max_time):
    cdef float start_time = time.clock()
    cdef float current_time = time.clock() - start_time
    cdef str empty_string = ""
    cdef MaxNode node, best_node
    cdef list node_heap = node_creator.get_start_nodes(y_length)
    cdef list solution_heap = []
    cdef list solutions, bounds
    cdef int max_depth = y_length - 1

    heapq.heapify(node_heap)
    best_node = MaxNode(empty_string, -150, 0, 0)
    while(len(node_heap) > 0 and current_time < max_time):
        node = heapq.heappop(node_heap)
        if  best_node < node:
            break
        node = depth_first_search(node, best_node, node_creator, node_heap, y_length, max_depth, alphabet)
        if node < best_node and len(node.y) == max_depth:
            best_node = add_children_to_solution_heap(node, best_node, node_creator, solution_heap, n_solutions,
                                                      y_length, alphabet)
        elif len(node.y) == y_length:
            best_node = add_node_to_solution_heap(node, best_node, solution_heap, n_solutions)
        current_time = time.clock() - start_time
    solutions, bounds = get_sorted_solutions_and_bounds(solution_heap)
    return solutions, bounds


@cython.boundscheck(False)
@cython.wraparound(False)
cdef MaxNode depth_first_search(MaxNode node, MaxNode best_node, NodeCreator node_creator, list heap,
                                int y_length, int max_depth, list alphabet):
    cdef int i
    cdef MaxNode parent_node, child
    cdef str letter, child_y
    for i in range(len(node.y), max_depth):
        parent_node = node
        child_y = alphabet[0] + parent_node.y
        node = node_creator.create_node(child_y, parent_node, y_length)
        for letter in alphabet[1:]:
            child_y = letter + parent_node.y
            child = node_creator.create_node(child_y, parent_node, y_length)
            if child < node:
                if node < best_node:
                     heapq.heappush(heap, node)
                node = child
            else:
                if(child < best_node):
                    heapq.heappush(heap, child)
        if best_node < node:
            break
    return node


@cython.boundscheck(False)
@cython.wraparound(False)
cdef MaxNode add_children_to_solution_heap(MaxNode parent_node, MaxNode best_node, NodeCreator node_creator,
                                           list solution_heap, int n_solutions, int y_length, list alphabet):
    cdef MaxNode solution_node
    cdef str letter, child_y

    for letter in alphabet:
        child_y = letter + parent_node.y
        solution_node = node_creator.create_node(child_y, parent_node, y_length)
        best_node = add_node_to_solution_heap(solution_node, best_node, solution_heap, n_solutions)
    return best_node


@cython.boundscheck(False)
@cython.wraparound(False)
cdef MaxNode add_node_to_solution_heap(MaxNode solution_node, MaxNode best_node, list solution_heap, int n_solutions):
    cdef MinNode solution_min_node
    if len(solution_heap) < n_solutions or solution_node < best_node:
        solution_min_node = solution_node.convert_to_min_node()
        heapq.heappush(solution_heap, solution_min_node)
    if len(solution_heap) > n_solutions:
        solution_min_node =  heapq.heappop(solution_heap)
        best_node = solution_min_node.convert_to_max_node()
    return best_node


@cython.boundscheck(False)
@cython.wraparound(False)
cdef get_sorted_solutions_and_bounds(list solution_heap):
    cdef int i
    cdef list bounds = []
    cdef list solutions = []
    cdef MinNode node
    for i in range(len(solution_heap)):
        node = heapq.heappop(solution_heap)
        solutions.append(node.y)
        bounds.append(node.get_bound())
    solutions.reverse()
    bounds.reverse()
    return solutions, bounds