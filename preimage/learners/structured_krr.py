__author__ = 'amelie'

import numpy
from scipy import linalg


class StructuredKernelRidgeRegression:
    def __init__(self, alpha, kernel, inference_model):
        self.alpha = alpha
        self.kernel = kernel
        self.inference_model = inference_model
        self.Weights_ = None
        self.X_train_ = None

    def fit(self, X, Y):
        Gram_matrix = self.kernel(X, X)
        self.Weights_ = self.__solve(Gram_matrix)
        self.X_train_ = X
        inference_parameters = InferenceFitParameters(self.Weights_, Gram_matrix, Y)
        self.inference_model.fit(inference_parameters)
        return self

    def __solve(self, Gram_matrix):
        # Faster to use fill diagonal than -> (Gram_matrix + self.alpha * numpy.eye(n_examples))
        diagonal = Gram_matrix.diagonal()
        numpy.fill_diagonal(Gram_matrix, diagonal + self.alpha)
        Weights = linalg.inv(Gram_matrix)
        numpy.fill_diagonal(Gram_matrix, diagonal)
        return Weights


# Basic structure for parameters inference_model.fit function.
# That way inference_models.fit(parameters) don't have unused parameters but only access the ones they need
class InferenceFitParameters:
    def __init__(self, Weights_, Gram_matrix, Y):
        self.Weights = Weights_
        self.Gram_matrix = Gram_matrix
        self.Y_train = Y