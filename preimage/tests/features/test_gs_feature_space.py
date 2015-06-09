__author__ = 'amelie'

from math import sqrt

import unittest2
import numpy.testing
from mock import patch
from scipy.sparse import csr_matrix

from preimage.features.gs_feature_space import GenericStringFeatureSpace


class TestGSFeatureSpace(unittest2.TestCase):
    def setUp(self):
        self.setup_parameters()
        self.setup_gs_weights()
        self.setup_feature_space()
        self.setup_position_weights_patch()
        self.setup_gs_kernel_patch()

    def setup_parameters(self):
        self.alphabet = ['a', 'b']
        self.abb = ['abb']
        self.abaaa = ['abaaa']
        self.abb_abaaa = self.abb + self.abaaa
        self.small_sigma = 1e-8
        self.medium_sigma = 0.5
        self.large_sigma = 1e8

    def setup_gs_weights(self):
        self.gs_weights_length_two_small_sigma_abb = [[0.5, 0], [0, 0.5]]
        self.gs_weights_length_four_small_sigma_abb = [[0.5, 0], [0, 0.5], [0, 0.5], [0, 0]]
        self.gs_weights_length_three_large_sigma_abb = [[0.5, 1], [0.5, 1], [0.5, 1]]
        self.gs_weights_length_three_medium_sigma_abb = [[0.5, 0.3], [0.25, 0.75], [0.05, 0.75]]
        self.gs_weights_length_three_small_sigma_abb_abaaa = [[0, 1.5, 0, 0], [0, 0, 1., 0.5], [1., 0, 0, 0]]

    def setup_feature_space(self):
        self.feature_space_builder_patch = patch('preimage.features.gs_feature_space.'
                                                 'build_feature_space_with_positions')
        self.feature_space_one_gram_abb = csr_matrix([[1., 0, 0, 1., 0, 1.]])
        self.feature_space_two_gram_abb_abaaa = csr_matrix([[0, 1., 0, 0, 0, 0, 0, 1., 0, 0, 0, 0, 0, 0, 0, 0],
                                                            [0, 1., 0, 0, 0, 0, 1., 0, 1., 0, 0, 0, 1., 0, 0, 0]])
        self.feature_space_normalized_one_abb = [[1. / sqrt(3), 0, 0, 1. / sqrt(3), 0, 1. / sqrt(3)]]
        self.feature_space_normalized_two_gram_abb_abaaa = [
            [0, 1. / sqrt(2), 0, 0, 0, 0, 0, 1. / sqrt(2), 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1. / sqrt(3), 0, 0, 0, 0, 1. / sqrt(3), 0, 1. / sqrt(3), 0, 0, 0,
             1. / sqrt(3), 0, 0, 0]]

    def setup_position_weights_patch(self):
        self.position_patch = patch('preimage.features.gs_feature_space.compute_position_weights')
        self.position_weights_length_two_small_sigma_abb = numpy.array([[1., 0, 0], [0., 1, 0]])
        self.position_weights_length_four_small_sigma_abb = numpy.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.], [0, 0, 0]])
        self.position_weights_length_three_large_sigma_abb = numpy.array([[1., 1., 1.], [1., 1, 1.], [1., 1., 1.]])
        self.position_weights_length_three_medium_sigma_abb = numpy.array([[1, 0.5, 0.1], [0.5, 1, 0.5], [0.1, 0.5, 1]])
        self.position_weights_length_four_small_sigma_abb_abaaa = numpy.array([[1., 0, 0, 0], [0, 1., 0, 0],
                                                                               [0, 0, 1., 0]])

    def setup_gs_kernel_patch(self):
        self.gs_kernel_patch = patch('preimage.features.gs_feature_space.element_wise_kernel')
        self.element_wise_kernel_one_gram_abb = [3.]
        self.element_wise_kernel_two_gram_abb_abaa = [2., 3.]


    def test_small_length_small_sigma_compute_one_gram_weights_returns_expected_weights(self):
        self.feature_space_builder_patch.start().return_value = self.feature_space_one_gram_abb
        self.position_patch.start().side_effect = self.position_weights_length_two_small_sigma_abb
        feature_space = GenericStringFeatureSpace(self.alphabet, n=1, Y=self.abb, sigma_position=self.small_sigma,
                                                  is_normalized=False)

        gs_weights = feature_space.compute_weights(y_weights=numpy.array([0.5]), y_length=2)

        numpy.testing.assert_array_equal(gs_weights, self.gs_weights_length_two_small_sigma_abb)

    def test_large_length_small_sigma_compute_one_gram_weights_returns_expected_weights(self):
        self.feature_space_builder_patch.start().return_value = self.feature_space_one_gram_abb
        self.position_patch.start().side_effect = self.position_weights_length_four_small_sigma_abb
        feature_space = GenericStringFeatureSpace(self.alphabet, n=1, Y=self.abb, sigma_position=self.small_sigma,
                                                  is_normalized=False)

        gs_weights = feature_space.compute_weights(y_weights=numpy.array([0.5]), y_length=4)

        numpy.testing.assert_array_equal(gs_weights, self.gs_weights_length_four_small_sigma_abb)

    def test_large_sigma_compute_one_gram_weights_returns_expected_weights(self):
        self.feature_space_builder_patch.start().return_value = self.feature_space_one_gram_abb
        self.position_patch.start().side_effect = self.position_weights_length_three_large_sigma_abb
        feature_space = GenericStringFeatureSpace(self.alphabet, n=1, Y=self.abb, sigma_position=self.large_sigma,
                                                  is_normalized=False)

        gs_weights = feature_space.compute_weights(y_weights=numpy.array([0.5]), y_length=3)

        numpy.testing.assert_array_equal(gs_weights, self.gs_weights_length_three_large_sigma_abb)

    def test_medium_sigma_compute_one_gram_weights_returns_expected_weights(self):
        self.feature_space_builder_patch.start().return_value = self.feature_space_one_gram_abb
        self.position_patch.start().side_effect = self.position_weights_length_three_medium_sigma_abb
        feature_space = GenericStringFeatureSpace(self.alphabet, n=1, Y=self.abb, sigma_position=self.medium_sigma,
                                                  is_normalized=False)

        gs_weights = feature_space.compute_weights(y_weights=numpy.array([0.5]), y_length=3)

        numpy.testing.assert_array_equal(gs_weights, self.gs_weights_length_three_medium_sigma_abb)

    def test_two_gram_two_y_small_sigma_compute_one_gram_weights_returns_expected_weights(self):
        self.feature_space_builder_patch.start().return_value = self.feature_space_two_gram_abb_abaaa
        self.position_patch.start().side_effect = self.position_weights_length_four_small_sigma_abb_abaaa
        feature_space = GenericStringFeatureSpace(self.alphabet, n=2, Y=self.abb_abaaa, sigma_position=self.small_sigma,
                                                  is_normalized=False)

        gs_weights = feature_space.compute_weights(y_weights=numpy.array([0.5, 1]), y_length=4)

        numpy.testing.assert_almost_equal(gs_weights, self.gs_weights_length_three_small_sigma_abb_abaaa)

    def test_compute_weights_does_not_change_feature_space(self):
        self.feature_space_builder_patch.start().return_value = self.feature_space_one_gram_abb
        self.position_patch.start().side_effect = self.position_weights_length_two_small_sigma_abb
        feature_space = GenericStringFeatureSpace(self.alphabet, n=1, Y=self.abb, sigma_position=self.small_sigma,
                                                  is_normalized=False)

        feature_space.compute_weights(y_weights=numpy.array([0.5]), y_length=2)
        gs_feature_space = feature_space._feature_space

        numpy.testing.assert_array_equal(gs_feature_space.toarray(), self.feature_space_one_gram_abb.toarray())

    def test_one_gram_one_y_small_sigma_normalize_feature_space_is_normalized(self):
        self.feature_space_builder_patch.start().return_value = self.feature_space_one_gram_abb
        self.gs_kernel_patch.start().return_value = self.element_wise_kernel_one_gram_abb

        feature_space = GenericStringFeatureSpace(self.alphabet, n=1, Y=self.abb, sigma_position=self.small_sigma,
                                                  is_normalized=True)
        normalized_feature_space = feature_space._feature_space.toarray()

        numpy.testing.assert_array_equal(normalized_feature_space, self.feature_space_normalized_one_abb)

    def test_two_gram_two_y_small_sigma_normalize_feature_space_is_normalized(self):
        self.feature_space_builder_patch.start().return_value = self.feature_space_two_gram_abb_abaaa
        self.gs_kernel_patch.start().return_value = self.element_wise_kernel_two_gram_abb_abaa

        feature_space = GenericStringFeatureSpace(self.alphabet, n=2, Y=self.abb, sigma_position=self.small_sigma,
                                                  is_normalized=True)
        normalized_feature_space = feature_space._feature_space.toarray()

        numpy.testing.assert_array_equal(normalized_feature_space, self.feature_space_normalized_two_gram_abb_abaaa)


if __name__ == '__main__':
    unittest2.main()