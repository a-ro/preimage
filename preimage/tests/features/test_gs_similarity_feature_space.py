__author__ = 'amelie'

from math import sqrt

import unittest2
import numpy.testing
from mock import Mock

from preimage.features.gs_similarity_feature_space import GenericStringSimilarityFeatureSpace


class TestGenericStringSimilarityFeatureSpace(unittest2.TestCase):
    def setUp(self):
        self.setup_parameters()
        self.setup_gs_weights()
        self.setup_gs_kernel_mock()

    def setup_parameters(self):
        self.alphabet = ['a', 'b']
        self.abb = ['abb']
        self.abaaa = ['abaaa']
        self.abb_abaaa = self.abb + self.abaaa

    def setup_gs_weights(self):
        self.gs_weights_length_two_small_sigma_p_abb = numpy.array([[1., 0.5], [0.5, 1.]])
        self.gs_weights_length_two_small_sigma_p_abb_abaaa = numpy.array([[1.5, 0.75], [0.75, 1.5]])
        self.gs_weights_normalized_abb_abaaa = numpy.array(
            [[1. / sqrt(3) + 0.5 / sqrt(5), 0.5 / sqrt(3) + 0.25 / sqrt(5)],
             [0.5 / sqrt(3) + 0.25 / sqrt(5), 1. / sqrt(3) + 0.5 / sqrt(5)]])
        self.gs_weights_length_four_small_sigma_p_abb = numpy.array([[1., 0.5], [0.5, 1.], [0.5, 1.], [0, 0]])
        self.gs_weights_length_two_large_sigma_p_abb = numpy.array([[2., 2.5], [2, 2.5]])
        self.gs_weights_two_gram_length_two_small_sigma_p_abb = numpy.array([[2., 3, 1.25, 2]])
        self.gs_weights_two_gram_length_three_small_sigma_p_abb = numpy.array([[1.5, 2, 0.75, 1], [1.25, 2, 2, 3]])

    def setup_gs_kernel_mock(self):
        self.positions_small_sigma = numpy.eye(5, dtype=numpy.float64)
        self.positions_large_sigma = numpy.ones((5, 5), dtype=numpy.float64)
        self.similarity_matrix = numpy.array([[1, 0.5], [0.5, 1]])
        self.gs_kernel_mock = Mock()
        self.gs_kernel_mock.get_position_matrix.return_value = self.positions_small_sigma
        self.gs_kernel_mock.get_alphabet_similarity_matrix.return_value = self.similarity_matrix
        self.gs_kernel_mock.element_wise_kernel.return_value = [3, 5]

    def test_small_length_small_sigma_p_compute_one_gram_weights_returns_expected_weights(self):
        feature_space = GenericStringSimilarityFeatureSpace(self.alphabet, n=1, Y=self.abb, is_normalized=False,
                                                            gs_kernel=self.gs_kernel_mock)

        gs_weights = feature_space.compute_weights(y_weights=numpy.array([1.]), y_length=2)

        numpy.testing.assert_array_equal(gs_weights, self.gs_weights_length_two_small_sigma_p_abb)

    def test_two_y_small_length_small_sigma_p_compute_one_gram_weights_returns_expected_weights(self):
        feature_space = GenericStringSimilarityFeatureSpace(self.alphabet, n=1, Y=self.abb_abaaa, is_normalized=False,
                                                            gs_kernel=self.gs_kernel_mock)

        gs_weights = feature_space.compute_weights(y_weights=numpy.array([1., 0.5]), y_length=2)

        numpy.testing.assert_array_equal(gs_weights, self.gs_weights_length_two_small_sigma_p_abb_abaaa)

    def test_two_y_normalized_small_sigma_p_compute_one_gram_weights_returns_expected_weights(self):
        feature_space = GenericStringSimilarityFeatureSpace(self.alphabet, n=1, Y=self.abb_abaaa, is_normalized=True,
                                                            gs_kernel=self.gs_kernel_mock)

        gs_weights = feature_space.compute_weights(y_weights=numpy.array([1., 0.5]), y_length=2)

        numpy.testing.assert_array_equal(gs_weights, self.gs_weights_normalized_abb_abaaa)

    def test_small_length_small_sigma_p_half_weight_compute_one_gram_weights_returns_expected_weights(self):
        feature_space = GenericStringSimilarityFeatureSpace(self.alphabet, n=1, Y=self.abb, is_normalized=False,
                                                            gs_kernel=self.gs_kernel_mock)

        gs_weights = feature_space.compute_weights(y_weights=numpy.array([0.5]), y_length=2)

        numpy.testing.assert_array_equal(gs_weights, 0.5 * self.gs_weights_length_two_small_sigma_p_abb)

    def test_large_length_small_sigma_p_compute_one_gram_weights_returns_expected_weights(self):
        feature_space = GenericStringSimilarityFeatureSpace(self.alphabet, n=1, Y=self.abb, is_normalized=False,
                                                            gs_kernel=self.gs_kernel_mock)

        gs_weights = feature_space.compute_weights(y_weights=numpy.array([1.]), y_length=4)

        numpy.testing.assert_array_equal(gs_weights, self.gs_weights_length_four_small_sigma_p_abb)

    def test_large_length_large_sigma_p_compute_one_gram_weights_returns_n_gram_weights(self):
        self.gs_kernel_mock.get_position_matrix.return_value = self.positions_large_sigma
        feature_space = GenericStringSimilarityFeatureSpace(self.alphabet, n=1, Y=self.abb, is_normalized=False,
                                                            gs_kernel=self.gs_kernel_mock)

        gs_weights = feature_space.compute_weights(y_weights=numpy.array([1.]), y_length=2)

        numpy.testing.assert_array_equal(gs_weights, self.gs_weights_length_two_large_sigma_p_abb)

    def test_small_sigma_p_compute_two_gram_weights_returns_expected_weights(self):
        feature_space = GenericStringSimilarityFeatureSpace(self.alphabet, n=2, Y=self.abb, is_normalized=False,
                                                            gs_kernel=self.gs_kernel_mock)

        gs_weights = feature_space.compute_weights(y_weights=numpy.array([1.]), y_length=2)

        numpy.testing.assert_array_equal(gs_weights, self.gs_weights_two_gram_length_two_small_sigma_p_abb)

    def test_length_three_small_sigma_p_compute_two_gram_weights_returns_expected_weights(self):
        feature_space = GenericStringSimilarityFeatureSpace(self.alphabet, n=2, Y=self.abb, is_normalized=False,
                                                            gs_kernel=self.gs_kernel_mock)

        gs_weights = feature_space.compute_weights(y_weights=numpy.array([1.]), y_length=3)

        numpy.testing.assert_array_equal(gs_weights, self.gs_weights_two_gram_length_three_small_sigma_p_abb)


if __name__ == '__main__':
    unittest2.main()