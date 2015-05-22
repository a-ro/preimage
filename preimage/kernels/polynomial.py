__author__ = 'amelie'

import numpy
from sklearn.base import BaseEstimator


class PolynomialKernel(BaseEstimator):
    def __init__(self, degree=2, bias=1., is_normalized=True):
        self.degree = degree
        self.bias = bias
        self.is_normalized = is_normalized

    def __call__(self, X_one, X_two):
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