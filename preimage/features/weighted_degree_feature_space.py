__author__ = 'amelie'

import numpy

from preimage.features.string_feature_space import build_feature_space_with_positions


class WeightedDegreeFeatureSpace:
    """Output feature space for the Weighted Degree kernel

    Creates a sparse matrix representation of the n-grams in each training string. The representation takes in account
    the positions of the n-grams in the strings, This is used to compute the weights of the graph during the inference
    phase.

    Attributes
    ----------
    n : int
        n-gram length
    max_n_gram_count : int
        The number of n-grams in the training string of highest length.
    feature_space : sparse matrix, shape = [n_samples, max_n_gram_count * len(alphabet)**n]
        Sparse matrix representation of the n-grams in each training string, where n_samples is the number of training
        samples.
    """
    def __init__(self, alphabet, n, Y, is_normalized):
        """Create the output feature space for the Weighted Degree kernel

        Parameters
        ----------
        alphabet : list
            list of letters
        n : int
            n-gram length
        Y : array, [n_samples, ]
            The training strings.
        is_normalized : bool
            True if the feature space should be normalized, False otherwise.
        """
        self.n = int(n)
        self._alphabet_n_gram_count = len(alphabet) ** n
        self.feature_space = build_feature_space_with_positions(alphabet, self.n, Y)
        self._normalize(is_normalized, self.feature_space)
        self.max_n_gram_count = self._get_max_n_gram_count(self._alphabet_n_gram_count, self.feature_space)

    def _get_max_n_gram_count(self, alphabet_n_gram_count, feature_space):
        n_columns = feature_space.shape[1]
        max_n_gram_count = int(n_columns / alphabet_n_gram_count)
        return max_n_gram_count

    def _normalize(self, is_normalized, feature_space):
        if is_normalized:
            y_normalization = 1. / numpy.sqrt(numpy.array(feature_space.sum(axis=1).reshape(1, -1))[0])
            data_normalization = y_normalization.repeat(numpy.diff(feature_space.indptr))
            feature_space.data *= data_normalization

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
        weighted_degree_weights : [len(alphabet)**n, y_n_gram_count * len(alphabet)**n]
            Weight of each n-gram at each position.
        """
        y_n_gram_count = y_length - self.n + 1
        data_copy = numpy.copy(self.feature_space.data)
        self.feature_space.data *= self._repeat_each_y_weight_by_y_column_count(y_weights)
        weights_vector = numpy.array(self.feature_space.sum(axis=0))[0]
        self.feature_space.data = data_copy
        weighted_degree_weights = self._get_weight_for_each_graph_partition(y_n_gram_count, weights_vector)
        return weighted_degree_weights

    def _get_weight_for_each_graph_partition(self, y_n_gram_count, weights_vector):
        weights_matrix = weights_vector.reshape(self.max_n_gram_count, -1)
        if y_n_gram_count <= self.max_n_gram_count:
            weights_matrix = weights_matrix[0:y_n_gram_count, :]
        else:
            zero_weight_partitions = numpy.zeros((y_n_gram_count - self.max_n_gram_count, self._alphabet_n_gram_count))
            weights_matrix = numpy.concatenate((weights_matrix, zero_weight_partitions), axis=0)
        return weights_matrix

    def _repeat_each_y_weight_by_y_column_count(self, y_weights):
        return y_weights.repeat(numpy.diff(self.feature_space.indptr))