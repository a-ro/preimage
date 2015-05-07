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
        Gram_matrix = (numpy.dot(X_one, X_two.T) + self.bias) ** self.degree
        if self.is_normalized:
            Gram_matrix = self._normalize_gram_matrix(X_one, X_two, Gram_matrix)
        return Gram_matrix

    def _normalize_gram_matrix(self, X_one, X_two, Gram_matrix):
        X_one_diagonal = self._compute_x_x_similarity(X_one)
        X_two_diagonal = self._compute_x_x_similarity(X_two)
        Gram_matrix = ((Gram_matrix / numpy.sqrt(X_one_diagonal)).T / numpy.sqrt(X_two_diagonal)).T
        return Gram_matrix

    def _compute_x_x_similarity(self, X):
        x_x_similarity = ((X*X).sum(axis=1) + self.bias) ** self.degree
        x_x_similarity = x_x_similarity.reshape(-1, 1)
        return x_x_similarity