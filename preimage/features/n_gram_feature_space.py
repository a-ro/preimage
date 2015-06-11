__author__ = 'amelie'

import numpy

from preimage.features.string_feature_space import build_feature_space_without_positions


class NGramFeatureSpace:
    """Output feature space for the N-Gram Kernel

    Creates a sparse matrix representation of the n-grams in each training string. This is used to compute the weights
    of the graph during the inference phase.

    Attributes
    ----------
    feature_space : sparse matrix, shape = [n_samples, len(alphabet)**n]
        Sparse matrix representation of the n-grams in each training string, where n_samples is the number of training
        samples.
    """

    def __init__(self, alphabet, n, Y, is_normalized):
        """Create the output feature space for the N-Gram Kernel

        Parameters
        ----------
        alphabet : list
            list of letters
        n : int
            N-gram length.
        Y : array, [n_samples, ]
            The training strings.
        is_normalized : bool
            True if the feature space should be normalized, False otherwise.
        """
        self.feature_space = build_feature_space_without_positions(alphabet, n, Y)
        self._normalize(is_normalized, self.feature_space)

    def _normalize(self, is_normalized, feature_space):
        if is_normalized:
            y_normalization = self._get_y_normalization(feature_space)
            data_normalization = y_normalization.repeat(numpy.diff(feature_space.indptr))
            feature_space.data *= data_normalization

    def _get_y_normalization(self, feature_space):
        y_normalization = (feature_space.multiply(feature_space)).sum(axis=1)
        y_normalization = 1. / numpy.sqrt(numpy.array((y_normalization.reshape(1, -1))[0]))
        return y_normalization

    def compute_weights(self, y_weights):
        """Compute the inference graph weights

        Parameters
        ----------
        y_weights :  array, [n_samples]
            Weight of each training example.

        Returns
        -------
        n_gram_weights : [len(alphabet)**n]
            Weight of each n-gram.
        """
        data_copy = numpy.copy(self.feature_space.data)
        self.feature_space.data *= self._repeat_each_y_weight_by_y_column_count(y_weights)
        n_gram_weights = numpy.array(self.feature_space.sum(axis=0))[0]
        self.feature_space.data = data_copy
        return n_gram_weights

    def _repeat_each_y_weight_by_y_column_count(self, y_weights):
        return y_weights.repeat(numpy.diff(self.feature_space.indptr))