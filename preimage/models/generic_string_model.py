__author__ = 'amelie'

from preimage.features.gs_feature_space import GenericStringFeatureSpace
from preimage.models.model import Model
from preimage.inference.graph_builder import GraphBuilder
from preimage.inference._branch_and_bound import branch_and_bound, branch_and_bound_no_length
from preimage.inference.bound_factory import get_gs_node_creator
from preimage.utils.alphabet import get_n_gram_to_index, get_n_grams


class GenericStringModel(Model):
    def __init__(self, alphabet, n, is_using_length=True, seed=42, max_time=30, sigma_position=1.):
        Model.__init__(self, alphabet, n, is_using_length)
        self._graph_builder = GraphBuilder(alphabet, n)
        self._is_normalized = True
        self._seed = seed
        self._max_time = max_time
        self._sigma_position = sigma_position
        self._n_grams = list(get_n_grams(alphabet, n))
        self._n_gram_to_index = get_n_gram_to_index(alphabet, n)

    def fit(self, inference_parameters):
        Model.fit(self, inference_parameters)
        self.feature_space_ = GenericStringFeatureSpace(self._alphabet, self._n, inference_parameters.Y_train,
                                                        self._sigma_position, self._is_normalized)

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
            graph = self._graph_builder.build_graph(n_gram_weights, y_length)
            node_creator = get_gs_node_creator(self._n, graph, n_gram_weights, y_length, self._n_gram_to_index,
                                                   self._n_grams, self._sigma_position)
            y_predicted, y_bound = branch_and_bound(node_creator, y_length, self._alphabet, self._max_time)
            Y_predictions.append(y_predicted)
        return Y_predictions

    def _predict_without_length(self, Y_weights):
        Y_predictions = []
        for y_weights in Y_weights:
            n_gram_weights = self.feature_space_.compute_weights(y_weights, self._max_length_)
            graph = self._graph_builder.build_graph(n_gram_weights, self._max_length_)
            node_creator = get_gs_node_creator(self._n, graph, n_gram_weights, self._max_length_,
                                                   self._n_gram_to_index, self._n_grams, self._sigma_position)
            y_predicted, y_bound = branch_and_bound_no_length(node_creator, self._min_length_, self._max_length_,
                                                              self._alphabet, self._max_time)
            Y_predictions.append(y_predicted)
        return Y_predictions