__author__ = 'amelie'

import numpy
from scipy import linalg
from sklearn.base import BaseEstimator


class StructuredKernelRidgeRegression(BaseEstimator):
    """Structured Kernel Ridge Regression.

    Attributes
    ----------
    alpha : float
        Regularization term.
    kernel : Callable
        Kernel function that computes the similarity between the samples.
    inference_model : Model
        Inference model used to solve the pre-image problem.
    weights_ : array, shape=[n_samples, n_samples]
        Learned weights, where n_samples is the number of training samples.
    X_train_ :  array, shape=[n_samples, n_features]
        Training samples.
    """
    def __init__(self, alpha, kernel, inference_model):
        self.alpha = alpha
        self.kernel = kernel
        self.inference_model = inference_model
        self.weights_ = None
        self.X_train_ = None

    def fit(self, X, Y, y_lengths=None):
        """Learn the weights.

        Parameters
        ----------
        X : array, shape=[n_samples, n_features]
            Training vectors, where n_samples is the number of samples and and n_features is the number of features
            in X.
        Y : array, shape=[n_samples, ]
            Target strings, where n_samples is the number of training samples.
        y_lengths : array, shape=[n_samples]
            Length of the training strings.

        Returns
        -------
        gram_matrix : array, shape = [n_samples_x1, n_samples_x2]
            Similarity of each string of X1 with each string of X2, n_samples_x1 is the number of samples in X1 and
            n_samples_x2 is the number of samples in X2.
        """
        gram_matrix = self.kernel(X, X)
        self.weights_ = self._solve(gram_matrix)
        self.X_train_ = X
        inference_parameters = InferenceFitParameters(self.weights_, gram_matrix, Y, y_lengths)
        self.inference_model.fit(inference_parameters)
        return self

    def _solve(self, gram_matrix):
        diagonal = numpy.copy(gram_matrix.diagonal())
        numpy.fill_diagonal(gram_matrix, diagonal + self.alpha)
        weights = linalg.inv(gram_matrix)
        numpy.fill_diagonal(gram_matrix, diagonal)
        return weights

    def predict(self, X, y_lengths=None):
        """Predict the target strings.

        Parameters
        ----------
        X : array, shape=[n_samples, n_features]
            Testing vectors, where n_samples is the number of samples and and n_features is the number of features
            in X.
        y_lengths : array, shape=[n_samples]
            Length of  the strings to predict, where n_samples is the number of testing samples.

        Returns
        -------
        Y_predicted : array, shape = [n_samples]
            Predicted strings, where n_samples  is the number of testing samples.
        """
        if self.weights_ is None:
            raise ValueError("The fit function must be called before predict")
        gram_matrix = self.kernel(self.X_train_, X)
        Y_weights = numpy.dot(self.weights_, gram_matrix).T
        Y_predicted = self.inference_model.predict(Y_weights, y_lengths)
        return Y_predicted


class InferenceFitParameters:
    """Parameters for the inference model.

    That way inference_model.fit(parameters) doesn't have unused parameters but only access the one it needs
    .
    Attributes
    ----------
    weights : array, shape = [n_samples, n_samples]
        Learned weights, where n_samples is the number of training samples.
    gram_matrix : array, shape = [n_samples, n_samples]
        Gram_matrix of the training samples.
    Y_train : array, shape = [n_samples, ]
        Training strings.
    y_lengths : array, shape = [n_samples]
        Length of each training string in Y_train.
    """
    def __init__(self, weights, gram_matrix, Y, y_lengths):
        self.weights = weights
        self.gram_matrix = gram_matrix
        self.Y_train = Y
        self.y_lengths = y_lengths