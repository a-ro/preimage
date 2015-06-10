__author__ = 'amelie'

import numpy

from preimage.features.string_feature_space import build_feature_space_with_positions
from preimage.utils.position import compute_position_weights
from preimage.kernels.generic_string import element_wise_kernel


class GenericStringFeatureSpace:
    """Output feature space for the Generic String kernel with position weights

    Creates a sparse matrix representation of the n-grams in each training string. The representation takes in account
    the positions of the n-grams in the strings, This is used to compute the weights of the graph during the inference
    phase. This doesn't take in account the similarity between the n-grams (no sigma_c).

    Attributes
    ----------
    n : int
        n-gram length
    sigma_position : float
        Parameter of the Generic String Kernel controlling the penalty incurred when two n-grams are not sharing the
        same position.
    max_n_gram_count : int
        The number of n-grams in the training string of highest length.
    feature_space : sparse matrix, shape = [n_samples, max_n_gram_count * len(alphabet)**n]
        Sparse matrix representation of the n-grams in each training string, where n_samples is the number of training
        samples.
    """

    def __init__(self, alphabet, n, Y, sigma_position, is_normalized):
        """Create the output feature space for the Generic String kernel

        Parameters
        ----------
        alphabet : list
            list of letters
        n : int
            n-gram length
        Y : array, [n_samples, ]
            The training strings.
        sigma_position : float
            Parameter of the Generic String Kernel controlling the penalty incurred when two n-grams are not sharing the
            same position.
        is_normalized : bool
            True if the feature space should be normalized, False otherwise.
        """
        self.n = int(n)
        self.sigma_position = sigma_position
        self._alphabet_n_gram_count = len(alphabet) ** n
        self.feature_space = build_feature_space_with_positions(alphabet, self.n, Y)
        self.max_n_gram_count = self._get_max_n_gram_count(self._alphabet_n_gram_count, self.feature_space)
        self._normalize(self.feature_space, self.n, Y, sigma_position, is_normalized, alphabet)

    def _get_max_n_gram_count(self, alphabet_n_gram_count, feature_space):
        n_columns = feature_space.shape[1]
        max_n_gram_count = int(n_columns / alphabet_n_gram_count)
        return max_n_gram_count

    def _normalize(self, feature_space, n, Y, sigma_position, is_normalized, alphabet):
        if is_normalized:
            y_y_similarity = element_wise_kernel(Y, sigma_position, n, alphabet)
            y_normalization = 1. / numpy.sqrt(y_y_similarity)
            data_normalization = y_normalization.repeat(numpy.diff(feature_space.indptr))
            feature_space.data *= data_normalization

    def _get_n_gram_count_in_each_y(self, n, Y):
        y_n_gram_counts = numpy.array([len(y) - n + 1 for y in Y])
        return y_n_gram_counts

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
            Weight of each n-gram at each position.
        """
        y_n_gram_count = y_length - self.n + 1
        data_copy = numpy.copy(self.feature_space.data)
        self.feature_space.data *= self._repeat_each_y_weight_by_y_column_count(y_weights)
        weighted_degree_weights = numpy.array(self.feature_space.sum(axis=0))[0].reshape(self.max_n_gram_count, -1)
        self.feature_space.data = data_copy
        gs_weights = self._transform_in_gs_weights(y_n_gram_count, weighted_degree_weights)
        return gs_weights

    def _transform_in_gs_weights(self, y_n_gram_count, weighted_degree_weights):
        gs_weights = numpy.empty((y_n_gram_count, self._alphabet_n_gram_count))
        for i in range(y_n_gram_count):
            position_weights = compute_position_weights(i, self.max_n_gram_count, self.sigma_position).reshape(-1, 1)
            gs_weights[i, :] = (weighted_degree_weights * position_weights).sum(axis=0)
        return gs_weights

    def _repeat_each_y_weight_by_y_column_count(self, y_weights):
        return y_weights.repeat(numpy.diff(self.feature_space.indptr))