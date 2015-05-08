__author__ = 'amelie'

import numpy
from scipy import linalg
from sklearn.base import BaseEstimator

class StructuredKernelRidgeRegression(BaseEstimator):
    def __init__(self, alpha, kernel, inference_model):
        self.alpha = alpha
        self.kernel = kernel
        self.inference_model = inference_model
        self.Weights_ = None
        self.X_train_ = None

    def fit(self, X, Y, y_lengths=None):
        Gram_matrix = self.kernel(X, X)
        self.Weights_ = self._solve(Gram_matrix)
        self.X_train_ = X
        inference_parameters = InferenceFitParameters(self.Weights_, Gram_matrix, Y, y_lengths)
        self.inference_model.fit(inference_parameters)
        return self

    def _solve(self, Gram_matrix):
        diagonal = Gram_matrix.diagonal()
        numpy.fill_diagonal(Gram_matrix, diagonal + self.alpha)
        Weights = linalg.inv(Gram_matrix)
        numpy.fill_diagonal(Gram_matrix, diagonal)
        return Weights

    def predict(self, X, y_lengths=None):
        if self.Weights_ is None:
            raise ValueError("The fit function must be called before predict")
        Gram_matrix = self.kernel(self.X_train_, X)
        Y_weights = numpy.dot(self.Weights_, Gram_matrix).T
        Y_predicted = self.inference_model.predict(Y_weights, y_lengths)
        return Y_predicted


# Basic structure for parameters inference_model.fit function.
# That way inference_models.fit(parameters) don't have unused parameters but only access the ones they need
class InferenceFitParameters:
    def __init__(self, Weights, Gram_matrix, Y, y_lengths):
        self.Weights = Weights
        self.Gram_matrix = Gram_matrix
        self.Y_train = Y
        self.y_lengths = y_lengths