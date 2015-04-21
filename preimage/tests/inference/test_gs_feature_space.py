__author__ = 'amelie'

import unittest2
import numpy.testing
from mock import patch
from scipy.sparse import csr_matrix

from preimage.inference.gs_feature_space import GenericStringFeatureSpace


class TestGSFeatureSpace(unittest2.TestCase):
    def setUp(self):
        self.setup_parameters()
        self.setup_gs_weights()
        self.setup_feature_space_patch()
        self.setup_position_weights_patch()

    def setup_parameters(self):
        self.alphabet = ['a', 'b']
        self.abb = ['abb']
        self.abaaa = ['abaaa']
        self.abb_abaaa = self.abb + self.abaaa
        self.small_sigma = 1e-8
        self.medium_sigma = 0.5
        self.large_sigma = 1e8

    def setup_gs_weights(self):
        self.GS_weights_length_two_small_sigma_abb = [[0.5, 0], [0, 0.5]]
        self.GS_weights_length_four_small_sigma_abb = [[0.5, 0], [0, 0.5], [0, 0.5], [0, 0]]
        self.GS_weights_length_three_large_sigma_abb = [[0.5, 1], [0.5, 1], [0.5, 1]]
        self.GS_weights_length_three_medium_sigma_abb = [[0.5, 0.3], [0.25, 0.75], [0.05, 0.75]]
        self.GS_weights_length_three_small_sigma_abb_abaaa = [[0, 1.5, 0, 0], [0, 0, 1., 0.5], [1., 0, 0, 0]]

    def setup_feature_space_patch(self):
        self.feature_space_builder_patch = patch('preimage.inference.gs_feature_space.'
                                                 'build_feature_space_with_positions')
        self.Feature_space_one_gram_abb = csr_matrix([[1., 0, 0, 1., 0, 1.]])
        self.Feature_space_two_gram_abb_abaaa = csr_matrix([[0, 1., 0, 0, 0, 0, 0, 1., 0, 0, 0, 0, 0, 0, 0, 0],
                                                            [0, 1., 0, 0, 0, 0, 1., 0, 1., 0, 0, 0, 1., 0, 0, 0]])

    def setup_position_weights_patch(self):
        self.position_patch = patch('preimage.inference.gs_feature_space.compute_position_weights')
        self.position_weights_length_two_small_sigma_abb = numpy.array([[1., 0, 0], [0., 1, 0]])
        self.position_weights_length_four_small_sigma_abb = numpy.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.], [0, 0, 0]])
        self.position_weights_length_three_large_sigma_abb = numpy.array([[1., 1., 1.], [1., 1, 1.], [1., 1., 1.]])
        self.position_weights_length_three_medium_sigma_abb = numpy.array([[1, 0.5, 0.1], [0.5, 1, 0.5], [0.1, 0.5, 1]])
        self.position_weights_length_four_small_sigma_abb_abaaa = numpy.array([[1., 0, 0, 0], [0, 1., 0, 0],
                                                                               [0, 0, 1., 0]])

    def test_small_length_small_sigma_compute_one_gram_weights_returns_expected_weights(self):
        self.feature_space_builder_patch.start().return_value = self.Feature_space_one_gram_abb
        self.position_patch.start().side_effect = self.position_weights_length_two_small_sigma_abb
        feature_space = GenericStringFeatureSpace(self.alphabet, n=1, Y=self.abb, sigma_position=self.small_sigma)

        GS_weights = feature_space.compute_weights(y_weights=numpy.array([0.5]), y_length=2)

        numpy.testing.assert_array_equal(GS_weights, self.GS_weights_length_two_small_sigma_abb)

    def test_large_length_small_sigma_compute_one_gram_weights_returns_expected_weights(self):
        self.feature_space_builder_patch.start().return_value = self.Feature_space_one_gram_abb
        self.position_patch.start().side_effect = self.position_weights_length_four_small_sigma_abb
        feature_space = GenericStringFeatureSpace(self.alphabet, n=1, Y=self.abb, sigma_position=self.small_sigma)

        GS_weights = feature_space.compute_weights(y_weights=numpy.array([0.5]), y_length=4)

        numpy.testing.assert_array_equal(GS_weights, self.GS_weights_length_four_small_sigma_abb)

    def test_large_sigma_compute_one_gram_weights_returns_expected_weights(self):
        self.feature_space_builder_patch.start().return_value = self.Feature_space_one_gram_abb
        self.position_patch.start().side_effect = self.position_weights_length_three_large_sigma_abb
        feature_space = GenericStringFeatureSpace(self.alphabet, n=1, Y=self.abb, sigma_position=self.large_sigma)

        GS_weights = feature_space.compute_weights(y_weights=numpy.array([0.5]), y_length=3)

        numpy.testing.assert_array_equal(GS_weights, self.GS_weights_length_three_large_sigma_abb)

    def test_medium_sigma_compute_one_gram_weights_returns_expected_weights(self):
        self.feature_space_builder_patch.start().return_value = self.Feature_space_one_gram_abb
        self.position_patch.start().side_effect = self.position_weights_length_three_medium_sigma_abb
        feature_space = GenericStringFeatureSpace(self.alphabet, n=1, Y=self.abb, sigma_position=self.medium_sigma)

        GS_weights = feature_space.compute_weights(y_weights=numpy.array([0.5]), y_length=3)

        numpy.testing.assert_array_equal(GS_weights, self.GS_weights_length_three_medium_sigma_abb)

    def test_two_gram_two_y_small_sigma_compute_one_gram_weights_returns_expected_weights(self):
        self.feature_space_builder_patch.start().return_value = self.Feature_space_two_gram_abb_abaaa
        self.position_patch.start().side_effect = self.position_weights_length_four_small_sigma_abb_abaaa
        feature_space = GenericStringFeatureSpace(self.alphabet, n=2, Y=self.abb_abaaa,
                                                  sigma_position=self.small_sigma)

        GS_weights = feature_space.compute_weights(y_weights=numpy.array([0.5, 1]), y_length=4)

        numpy.testing.assert_almost_equal(GS_weights, self.GS_weights_length_three_small_sigma_abb_abaaa)

    def test_compute_weights_does_not_change_feature_space(self):
        self.feature_space_builder_patch.start().return_value = self.Feature_space_one_gram_abb
        self.position_patch.start().side_effect = self.position_weights_length_two_small_sigma_abb
        feature_space = GenericStringFeatureSpace(self.alphabet, n=1, Y=self.abb, sigma_position=self.small_sigma)

        feature_space.compute_weights(y_weights=numpy.array([0.5]), y_length=2)
        GS_feature_space = feature_space._Feature_space

        numpy.testing.assert_array_equal(GS_feature_space.toarray(), self.Feature_space_one_gram_abb.toarray())


if __name__ == '__main__':
    unittest2.main()