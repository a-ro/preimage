__author__ = 'amelie'

from preimage.features.weighted_degree_feature_space import WeightedDegreeFeatureSpace
from preimage.inference.graph_builder import GraphBuilder
from preimage.models.model import Model


class WeightedDegreeModel(Model):
    def __init__(self, alphabet, n, is_using_length=True):
        self._graph_builder = GraphBuilder(alphabet, n)
        self._is_normalized = True
        Model.__init__(self, alphabet, n, is_using_length)

    def fit(self, inference_parameters):
        Model.fit(self, inference_parameters)
        self.feature_space_ = WeightedDegreeFeatureSpace(self._alphabet, self._n, inference_parameters.Y_train,
                                                         self._is_normalized)

    def predict(self, Y_weights, y_lengths):
        if self._is_using_length:
            self._verify_y_lengths_is_not_none_when_use_length(y_lengths)
            Y_predictions = self._predict_with_length(Y_weights, y_lengths)
        else:
            Y_predictions = self._predict_without_length(Y_weights)
        return Y_predictions

    def _predict_with_length(self, Y_weights, y_lengths):
        Y_predictions = []
        for y_weights, y_length in zip(Y_weights, y_lengths):
            n_gram_weights = self.feature_space_.compute_weights(y_weights, y_length)
            y_predicted = self._graph_builder.find_max_string(n_gram_weights, y_length)
            Y_predictions.append(y_predicted)
        return Y_predictions

    def _predict_without_length(self, Y_weights):
        Y_predictions = []
        for y_weights in Y_weights:
            n_gram_weights = self.feature_space_.compute_weights(y_weights, self._max_length_)
            y_predicted = self._graph_builder.find_max_string_in_length_range(n_gram_weights, self._min_length_,
                                                                             self._max_length_, self._is_normalized)
            Y_predictions.append(y_predicted)
        return Y_predictions