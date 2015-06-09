# cython: profile=True

import cython
import heapq
import numpy
cimport numpy
import time
from preimage.inference.node cimport MaxNode
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
        node = depth_first_search(node, best_node, node_creator, heap, y_length, alphabet)
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
            node = depth_first_search(node, best_node, node_creator, heap, length, alphabet)
            if node < best_node and len(node.y) == length:
                best_node = node
    return best_node


@cython.boundscheck(False)
@cython.wraparound(False)
cdef MaxNode depth_first_search(MaxNode node, MaxNode best_node, NodeCreator node_creator, list heap,
                                int y_length, list alphabet):
    cdef int i
    cdef MaxNode parent_node, child
    cdef str letter, child_y

    for i in range(len(node.y), y_length):
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