"""Factory initializing bounds and node creator for the branch and bound search."""

__author__ = 'amelie'

import numpy

from preimage.inference.bound_calculator import OCRMinBoundCalculator, MaxBoundCalculator, PeptideMinBoundCalculator
from preimage.inference.node_creator import NodeCreator
from preimage.utils.position import compute_position_weights
from preimage.utils.alphabet import get_n_gram_to_index, get_n_grams


def get_n_gram_node_creator(n, graph, graph_weights, y_length, n_gram_to_index, n_grams):
    """Create the bounds and the node creator for the branch and bound search of the n-gram kernel

    Parameters
    ----------
    n : int
        N-gram length.
    graph : array, shape = [n_partitions, len(alphabet)**n]
        Array representation of the graph. graph[i, j] represents the maximum value of a string of length i + n ending
        with the jth n-gram.
    graph_weights : array, shape = [len(alphabet)**n]
        Weight of each n-gram.
    y_length : int
        Length of the string to predict.
    n_gram_to_index : dict
        Dictionary of n-grams and their corresponding index.
    n_grams : list
        List of n-grams.

    Returns
    -------
    node_creator : NodeCreator
        Node creator for the branch and bound search instantiated with the n-gram bounds
    """
    max_bound = MaxBoundCalculator(n, graph, graph_weights.reshape(1, -1), n_gram_to_index)
    min_bound = OCRMinBoundCalculator(n, numpy.ones(y_length), n_grams)
    node_creator = NodeCreator(min_bound, max_bound, n_grams)
    return node_creator


def get_gs_node_creator(n, graph, graph_weights, y_length, n_gram_to_index, n_grams, sigma_position):
    """Create the bounds and the node creator for the branch and bound search of the generic string kernel.

    Only takes in account the position penalties when comparing strings, no n-gram similarity (no sigma_c).

    Parameters
    ----------
    n : int
        N-gram length.
    graph : array, shape = [n_partitions, len(alphabet)**n]
        Array representation of the graph. graph[i, j] represents the maximum value of a string of length i + n ending
        with the jth n-gram.
    graph_weights : array, shape = [n_partitions, len(alphabet)**n]
        Weight of each n-gram.
    y_length : int
        Length of the string to predict.
    n_gram_to_index : dict
        Dictionary of n-grams and their corresponding index.
    n_grams : list
        List of n-grams.
    sigma_position : float
        Parameter of the Generic String Kernel controlling the penalty incurred when two n-grams are not sharing the
        same position.

    Returns
    -------
    node_creator : NodeCreator
        Node creator for the branch and bound search instantiated with the generic string bounds
    """
    position_weights = compute_position_weights(0, y_length, sigma_position)
    max_bound = MaxBoundCalculator(n, graph, graph_weights, n_gram_to_index)
    min_bound = OCRMinBoundCalculator(n, position_weights, n_grams)
    node_creator = NodeCreator(min_bound, max_bound, n_grams)
    return node_creator


def get_gs_similarity_node_creator(alphabet, n, graph, graph_weights, y_length, gs_kernel):
    """Create the bounds and the node creator for the branch and bound search of the generic string kernel.

    Takes in account the position and the n-gram penalties when comparing strings (sigma_p and sigma_c in the
    gs kernel).

    Parameters
    ----------
    alphabet : list
        List of letters.
    n : int
        N-gram length.
    graph : array, shape = [n_partitions, len(alphabet)**n]
        Array representation of the graph. graph[i, j] represents the maximum value of a string of length i + n ending
        with the jth n-gram.
    graph_weights : array, shape = [n_partitions, len(alphabet)**n]
        Weight of each n-gram.
    y_length : int
        Length of the string to predict.
    gs_kernel : GenericStringKernel
        Generic String Kernel with position and n-gram penalties.

    Returns
    -------
    node_creator : NodeCreator
        Node creator for the branch and bound search instantiated with the generic string bounds
    """
    n_gram_to_index = get_n_gram_to_index(alphabet, n)
    letter_to_index = get_n_gram_to_index(alphabet, 1)
    n_grams = get_n_grams(alphabet, n)
    min_bound = PeptideMinBoundCalculator(n, len(alphabet), n_grams, letter_to_index, y_length, gs_kernel)
    max_bound = MaxBoundCalculator(n, graph, graph_weights, n_gram_to_index)
    node_creator = NodeCreator(min_bound, max_bound, n_grams)
    return node_creator