__author__ = 'amelie'

import numpy


class PolynomialKernel:
    def __init__(self, degree=2, bias=1., is_normalized=True):
        self.degree = degree
        self.bias = bias
        self.is_normalized = is_normalized

    def __call__(self, X_one, X_two):
        X_one = numpy.array(X_one)
        X_two = numpy.array(X_two)
        Gram_matrix = numpy.array(numpy.dot(X_one, X_two.T) + self.bias)
        Gram_matrix **= self.degree
        if self.is_normalized:
            Gram_matrix = self._normalize_gram_matrix(X_one, X_two, Gram_matrix)
        return Gram_matrix

    def _normalize_gram_matrix(self, X_one, X_two, Gram_matrix):
        X_one_diagonal = self.pairwise_similarity(X_one)
        X_two_diagonal = self.pairwise_similarity(X_two)
        Gram_matrix = ((Gram_matrix / numpy.sqrt(X_one_diagonal)).T / numpy.sqrt(X_two_diagonal)).T
        return Gram_matrix

    def pairwise_similarity(self, X):
        diagonal = numpy.empty((X.shape[0], 1))
        for i in range(X.shape[0]):
            diagonal[i] = numpy.dot(X[i], X[i].T) + self.bias
        diagonal **= self.degree
        return diagonal