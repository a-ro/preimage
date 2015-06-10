__author__ = 'amelie'

import numpy
from preimage.features.gs_similarity_weights import compute_gs_similarity_weights
from preimage.utils.alphabet import transform_strings_to_integer_lists, get_n_grams


# Shouldn't label this as "feature-space" since we don't use a sparse matrix representation here.
class GenericStringSimilarityFeatureSpace:
    """Output space for the Generic String kernel with position and n-gram similarity.

    Doesn't use a sparse matrix representation because it takes in account the similarity between the n-grams.
    This is used to compute the weights of the graph during the inference phase.

    Attributes
    ----------
    n : int
        n-gram length.
    is_normalized : bool
        True if the feature space should be normalized, False otherwise.
    max_train_length : int
        Length of the longest string in the training dataset.
    gs_kernel : GenericStringKernel
        Generic string kernel.
    """

    def __init__(self, alphabet, n, Y, is_normalized, gs_kernel):
        self.n = int(n)
        self.is_normalized = is_normalized
        self._y_lengths = numpy.array([len(y) for y in Y])
        self.max_train_length = numpy.max(self._y_lengths)
        self.gs_kernel = gs_kernel
        self._Y_int = transform_strings_to_integer_lists(Y, alphabet)
        self._n_grams_int = transform_strings_to_integer_lists(get_n_grams(alphabet, n), alphabet)
        self._n_gram_similarity_matrix = gs_kernel.get_alphabet_similarity_matrix()
        if is_normalized:
            self._normalization = numpy.sqrt(gs_kernel.element_wise_kernel(Y))

    def compute_weights(self, y_weights, y_length):
        """Compute the inference graph weights

        Parameters
        ----------
        y_weights :  array, [n_samples]
            Weight of each training example.
        y_length : int
            Length of the string to predict.

        Returns
        -------
        gs_weights : [len(alphabet)**n, y_n_gram_count * len(alphabet)**n]
            Weight of each n-gram at each position, where y_n_gram_count is the number of n-gram in y_length.
        """
        normalized_weights = numpy.copy(y_weights)
        max_length = max(y_length, self.max_train_length)
        if self.is_normalized:
            normalized_weights *= 1. / self._normalization
        n_partitions = y_length - self.n + 1
        position_matrix = self.gs_kernel.get_position_matrix(max_length)
        gs_weights = compute_gs_similarity_weights(n_partitions, self._n_grams_int, self._Y_int, normalized_weights,
                                                   self._y_lengths, position_matrix, self._n_gram_similarity_matrix)
        return numpy.array(gs_weights)