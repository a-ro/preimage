__author__ = 'amelie'

import numpy
from sklearn.base import BaseEstimator


class PolynomialKernel(BaseEstimator):
    """Polynomial kernel.

    Attributes
    ----------
    degree : int
        Degree.
    bias : float
        Bias.
    is_normalized : bool
        True if the kernel should be normalized, False otherwise.
    """
    def __init__(self, degree=2, bias=1., is_normalized=True):
        self.degree = degree
        self.bias = bias
        self.is_normalized = is_normalized

    def __call__(self, X_one, X_two):
        """Compute the similarity of all the vectors in X1 with all the vectors in X2.

        Parameters
        ----------
        X1 : array, shape=[n_samples, n_features]
            Vectors, where n_samples is the number of samples in X1 and n_features is the number of features.
        X2 : array, shape=[n_samples, n_features]
            Vectors, where n_samples is the number of samples in X2 and n_features is the number of features.

        Returns
        -------
        gram_matrix : array, shape = [n_samples_x1, n_samples_x2]
            Similarity of each vector of X1 with each vector of X2, where n_samples_x1 is the number of samples in X1
            and n_samples_x2 is the number of samples in X2.
        """
        X_one = numpy.array(X_one)
        X_two = numpy.array(X_two)
        gram_matrix = (numpy.dot(X_one, X_two.T) + self.bias) ** self.degree
        if self.is_normalized:
            gram_matrix = self._normalize_gram_matrix(X_one, X_two, gram_matrix)
        return gram_matrix

    def _normalize_gram_matrix(self, X_one, X_two, gram_matrix):
        x_one_diagonal = self._compute_element_wise_similarity(X_one)
        x_two_diagonal = self._compute_element_wise_similarity(X_two)
        gram_matrix = ((gram_matrix / numpy.sqrt(x_one_diagonal)).T / numpy.sqrt(x_two_diagonal)).T
        return gram_matrix

    def _compute_element_wise_similarity(self, X):
        x_x_similarity = ((X * X).sum(axis=1) + self.bias) ** self.degree
        x_x_similarity = x_x_similarity.reshape(-1, 1)
        return x_x_similarity