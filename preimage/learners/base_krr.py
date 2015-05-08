__author__ = 'amelie'

import numpy
from scipy import linalg
from sklearn.base import BaseEstimator

class KernelRidgeRegression(BaseEstimator):
    def __init__(self, alpha, kernel):
        self.alpha = alpha
        self.kernel = kernel

    def _solve(self, Gram_matrix):
        diagonal = Gram_matrix.diagonal()
        numpy.fill_diagonal(Gram_matrix, diagonal + self.alpha)
        Weights = linalg.inv(Gram_matrix)
        numpy.fill_diagonal(Gram_matrix, diagonal)
        return Weights