__author__ = 'amelie'

import numpy

from preimage.inference.euler import EulerianPath
from preimage.features.n_gram_feature_space import NGramFeatureSpace
from preimage.models.model import Model
from preimage.inference.graph_builder import GraphBuilder
from preimage.utils.alphabet import get_n_gram_to_index, get_n_grams


class EulerianPathModel(Model):
    def __init__(self, alphabet, n, is_using_length=True, seed=42):
        Model.__init__(self, alphabet, n, is_using_length)
        self._graph_builder = GraphBuilder(alphabet, n)
        self._is_normalized = False
        self._is_merging_path = True
        self._seed = seed
        self._n_grams = list(get_n_grams(alphabet, n))
        self._n_gram_to_index = get_n_gram_to_index(alphabet, n)
        self._thresholds_ = None

    def fit(self, inference_parameters):
        Model.fit(self, inference_parameters)
        self._feature_space_ = NGramFeatureSpace(self._alphabet, self._n, inference_parameters.Y_train,
                                                 self._is_normalized)
        if not self._is_using_length:
            Y_weights = numpy.dot(inference_parameters.weights, inference_parameters.gram_matrix).T
            self._find_thresholds(Y_weights)

    def _find_thresholds(self, Y_weights):
        n_examples = Y_weights.shape[0]
        Y_n_gram_weights = self._get_n_gram_weights(Y_weights, n_examples)
        n_gram_counts = self._feature_space_.compute_weights(numpy.ones(n_examples))
        n_gram_counts = numpy.array(n_gram_counts, dtype=numpy.int)
        self._thresholds_ = self._find_weights_where_sum_weights_above_is_n_gram_count(n_gram_counts, Y_n_gram_weights)

    def _get_n_gram_weights(self, Y_weights, n_training_examples):
        Y_n_gram_weights = numpy.empty((n_training_examples, len(self._alphabet) ** self._n))
        for y_index, y_weight in enumerate(Y_weights):
            Y_n_gram_weights[y_index] = self._feature_space_.compute_weights(y_weight)
        return Y_n_gram_weights

    def _find_weights_where_sum_weights_above_is_n_gram_count(self, n_gram_counts, Y_n_gram_weights):
        thresholds = numpy.zeros(len(self._alphabet) ** self._n)
        for n_gram_index, n_gram_count in enumerate(n_gram_counts):
            if n_gram_count > 0:
                n_gram_weights = Y_n_gram_weights[:, n_gram_index]
                threshold_index = numpy.argpartition(-n_gram_weights, n_gram_count)[n_gram_count]
                thresholds[n_gram_index] = n_gram_weights[threshold_index]
        return thresholds

    def predict(self, Y_weights, y_lengths):
        if self._is_using_length:
            self._verify_y_lengths_is_not_none_when_use_length(y_lengths)
            Y_predictions = self._predict_with_length(Y_weights, y_lengths)
        else:
            Y_predictions = self._predict_without_length(Y_weights)
        return Y_predictions

    def _predict_with_length(self, Y_weights, y_lengths):
        Y_predictions = []
        eulerian_path = EulerianPath(self._alphabet, self._n, self._min_length_, self._is_merging_path)
        for y_weights, y_length in zip(Y_weights, y_lengths):
            n_gram_weights = self._feature_space_.compute_weights(y_weights)
            y_predicted = eulerian_path.find_eulerian_path(n_gram_weights, y_length=y_length)
            Y_predictions.append(y_predicted)
        return Y_predictions

    def _predict_without_length(self, Y_weights):
        Y_predictions = []
        eulerian_path = EulerianPath(self._alphabet, self._n, self._min_length_, self._is_merging_path)
        for y_weights in Y_weights:
            n_gram_weights = self._feature_space_.compute_weights(y_weights)
            y_predicted = eulerian_path.find_eulerian_path(n_gram_weights, thresholds=self._thresholds_)
            Y_predictions.append(y_predicted)
        return Y_predictions