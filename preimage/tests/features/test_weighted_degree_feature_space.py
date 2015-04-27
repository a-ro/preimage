__author__ = 'amelie'

import unittest2
import numpy.testing
from mock import patch
from scipy.sparse import csr_matrix
from math import sqrt

from preimage.features.weighted_degree_feature_space import WeightedDegreeFeatureSpace


class TestWeightedDegreeFeatureSpace(unittest2.TestCase):
    def setUp(self):
        self.alphabet = ['a', 'b']
        self.abb = ['abb']
        self.abaaa = ['abaaa']
        self.abb_abaaa = self.abb + self.abaaa
        self.Feature_space_one_gram_abb = csr_matrix([[1., 0, 0, 1., 0, 1.]])
        self.Feature_space_two_gram_abb = csr_matrix([[0, 1., 0, 0, 0, 0, 0, 1.]])
        self.Feature_space_one_gram_abb_abaaa = csr_matrix([[1., 0, 0, 1., 0, 1., 0, 0, 0, 0],
                                                            [1., 0, 0, 1., 1., 0, 1., 0, 1., 0]])
        self.Feature_space_two_gram_abb_abaaa = csr_matrix([[0, 1., 0, 0, 0, 0, 0, 1., 0, 0, 0, 0, 0, 0, 0, 0],
                                                            [0, 1., 0, 0, 0, 0, 1., 0, 1., 0, 0, 0, 1., 0, 0, 0]])
        self.Feature_space_normalized_one_gram_abb = numpy.array([[1./sqrt(3), 0, 0, 1./sqrt(3), 0, 1./sqrt(3)]])
        self.Feature_space_normalized_two_gram_abb = numpy.array([[0, 1./sqrt(2), 0, 0, 0, 0, 0, 1./sqrt(2)]])
        self.Feature_space_normalized_one_gram_abb_abaaa = [[1./sqrt(3), 0, 0, 1./sqrt(3), 0, 1./sqrt(3), 0, 0, 0, 0],
                                                            [1./sqrt(5), 0, 0, 1./sqrt(5), 1./sqrt(5), 0, 1./sqrt(5), 0,
                                                             1./sqrt(5), 0]]
        self.Weighted_degree_weights_length_two_one_half_abb = [[0.5, 0], [0, 0.5]]
        self.Weighted_degree_weights_length_four_one_half_abb = [[0.5, 0], [0, 0.5], [0, 0.5], [0, 0]]
        self.Weighted_degree_weights_two_grams_abb_abaaa = [[0, 0.7, 0, 0], [0, 0, 0.5, 0.2], [0.5, 0, 0, 0],
                                                            [0.5, 0, 0, 0]]
        self.feature_space_builder_patch = patch('preimage.features.weighted_degree_feature_space.'
                                                 'build_feature_space_with_positions')

    def test_one_gram_one_y_normalized_feature_space_is_normalized(self):
        self.feature_space_builder_patch.start().return_value = self.Feature_space_one_gram_abb

        feature_space = WeightedDegreeFeatureSpace(self.alphabet, n=1, Y=self.abb, is_normalized=True)
        Feature_space_normalized = numpy.array(feature_space._Feature_space.todense())

        numpy.testing.assert_array_equal(Feature_space_normalized, self.Feature_space_normalized_one_gram_abb)

    def test_two_gram_one_y_normalized_feature_space_is_normalized(self):
        self.feature_space_builder_patch.start().return_value = self.Feature_space_two_gram_abb

        feature_space = WeightedDegreeFeatureSpace(self.alphabet, n=2, Y=self.abb, is_normalized=True)
        Feature_space_normalized = numpy.array(feature_space._Feature_space.todense())

        numpy.testing.assert_array_equal(Feature_space_normalized, self.Feature_space_normalized_two_gram_abb)

    def test_two_gram_two_y_normalized_feature_space_is_normalized(self):
        self.feature_space_builder_patch.start().return_value = self.Feature_space_one_gram_abb_abaaa

        feature_space = WeightedDegreeFeatureSpace(self.alphabet, n=1, Y=self.abb_abaaa, is_normalized=True)
        Feature_space_normalized = numpy.array(feature_space._Feature_space.todense())

        numpy.testing.assert_array_equal(Feature_space_normalized, self.Feature_space_normalized_one_gram_abb_abaaa)

    def test_smaller_length_than_train_y_length_compute_one_gram_weights_returns_expected_weights(self):
        self.feature_space_builder_patch.start().return_value = self.Feature_space_one_gram_abb
        feature_space = WeightedDegreeFeatureSpace(self.alphabet, n=1, Y=self.abb, is_normalized=False)

        Weighted_degree_weights = feature_space.compute_weights(y_weights=numpy.array([0.5]), y_length=2)

        numpy.testing.assert_array_equal(Weighted_degree_weights, self.Weighted_degree_weights_length_two_one_half_abb)

    def test_larger_length_than_train_y_length_compute_one_gram_weights_returns_expected_weights(self):
        self.feature_space_builder_patch.start().return_value = self.Feature_space_one_gram_abb
        feature_space = WeightedDegreeFeatureSpace(self.alphabet, n=1, Y=self.abb, is_normalized=False)

        Weighted_degree_weights = feature_space.compute_weights(y_weights=numpy.array([0.5]), y_length=4)

        numpy.testing.assert_array_equal(Weighted_degree_weights, self.Weighted_degree_weights_length_four_one_half_abb)

    def test_two_y_with_different_weights_compute_two_gram_weights_returns_expected_weights(self):
        self.feature_space_builder_patch.start().return_value = self.Feature_space_two_gram_abb_abaaa
        feature_space = WeightedDegreeFeatureSpace(self.alphabet, n=2, Y=self.abb_abaaa, is_normalized=False)

        Weighted_degree_weights = feature_space.compute_weights(y_weights=numpy.array([0.2, 0.5]), y_length=5)

        numpy.testing.assert_array_equal(Weighted_degree_weights, self.Weighted_degree_weights_two_grams_abb_abaaa)


    def test_compute_weights_does_not_change_feature_space(self):
        self.feature_space_builder_patch.start().return_value = self.Feature_space_one_gram_abb
        feature_space = WeightedDegreeFeatureSpace(self.alphabet, n=1, Y=self.abb, is_normalized=False)

        feature_space.compute_weights(y_weights=numpy.array([0.5]), y_length=2)
        Weighted_degree_feature_space = feature_space._Feature_space

        numpy.testing.assert_array_equal(Weighted_degree_feature_space.toarray(),
                                         self.Feature_space_one_gram_abb.toarray())


if __name__ == '__main__':
    unittest2.main()