__author__ = 'amelie'

import abc

from sklearn.base import BaseEstimator
import numpy

from preimage.exceptions.n_gram import NoYLengthsError


class Model(BaseEstimator):
    __metaclass__ = abc.ABCMeta

    def __init__(self, alphabet, n, is_using_length=True):
        self._n = n
        self._alphabet = alphabet
        self._is_using_length = is_using_length
        self._feature_space_ = None
        self._min_length_ = None
        self._max_length_ = None

    def fit(self, inference_parameters):
        self._find_min_max_length(inference_parameters.y_lengths, inference_parameters.Y_train)

    def _find_min_max_length(self, y_lengths, Y):
        if y_lengths is None:
            y_lengths = numpy.array([len(y) for y in Y])
        self._min_length_ = numpy.min(y_lengths)
        self._max_length_ = numpy.max(y_lengths)

    @abc.abstractclassmethod
    def predict(self, Y_weights, y_lengths):
        return

    def _verify_y_lengths_is_not_none_when_use_length(self, y_lengths):
        if self._is_using_length and y_lengths is None:
            raise NoYLengthsError()