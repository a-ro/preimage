__author__ = 'amelie'

import numpy
from scipy import linalg
from sklearn.base import BaseEstimator


class StructuredKernelRidgeRegression(BaseEstimator):
    def __init__(self, alpha, kernel, inference_model):
        self.alpha = alpha
        self.kernel = kernel
        self.inference_model = inference_model
        self.weights_ = None
        self.X_train_ = None

    def fit(self, X, Y, y_lengths=None):
        gram_matrix = self.kernel(X, X)
        self.weights_ = self._solve(gram_matrix)
        self.X_train_ = X
        inference_parameters = InferenceFitParameters(self.weights_, gram_matrix, Y, y_lengths)
        self.inference_model.fit(inference_parameters)
        return self

    def predict(self, X, y_lengths=None):
        if self.weights_ is None:
            raise ValueError("The fit function must be called before predict")
        gram_matrix = self.kernel(self.X_train_, X)
        Y_weights = numpy.dot(self.weights_, gram_matrix).T
        Y_predicted = self.inference_model.predict(Y_weights, y_lengths)
        return Y_predicted

    def _solve(self, gram_matrix):
        diagonal = numpy.copy(gram_matrix.diagonal())
        numpy.fill_diagonal(gram_matrix, diagonal + self.alpha)
        weights = linalg.inv(gram_matrix)
        numpy.fill_diagonal(gram_matrix, diagonal)
        return weights


# Basic structure for parameters inference_model.fit function.
# That way inference_models.fit(parameters) don't have unused parameters but only access the ones they need
class InferenceFitParameters:
    def __init__(self, weights, gram_matrix, Y, y_lengths):
        self.weights = weights
        self.gram_matrix = gram_matrix
        self.Y_train = Y
        self.y_lengths = y_lengths