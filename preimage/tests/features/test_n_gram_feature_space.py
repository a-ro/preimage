__author__ = 'amelie'

from math import sqrt

import unittest2
import numpy.testing
from mock import patch
from scipy.sparse import csr_matrix

from preimage.features.n_gram_feature_space import NGramFeatureSpace


class TestNGramFeatureSpace(unittest2.TestCase):
    def setUp(self):
        self.setup_alphabet()
        self.setup_feature_space()
        self.setup_weights()
        self.setup_patch()

    def setup_alphabet(self):
        self.alphabet = ['a', 'b']
        self.abb = ['abb']
        self.abaaa = ['abaaa']
        self.abb_abaaa = self.abb + self.abaaa

    def setup_feature_space(self):
        self.feature_space_one_gram_abb = csr_matrix([[1., 2.]])
        self.feature_space_two_gram_abb = csr_matrix([[0, 1., 0, 1.]])
        self.feature_space_two_gram_abb_abaaa = csr_matrix([[0, 1., 0, 1.], [2., 1., 1., 0]])
        self.feature_space_one_gram_abb_abaaa = csr_matrix([[1., 2.], [4., 1.]])
        self.feature_space_normalized_one_gram_abb = [[1. / sqrt(5), 2. / sqrt(5)]]
        self.feature_space_normalized_two_gram_abb = [[0, 1. / sqrt(2), 0, 1. / sqrt(2)]]
        self.feature_space_normalized_one_gram_abb_abaaa = [[1. / sqrt(5), 2. / sqrt(5)],
                                                            [4. / sqrt(17), 1. / sqrt(17)]]

    def setup_weights(self):
        self.one_gram_weights_one_half_abb = [0.5, 1]
        self.two_gram_weights_one_half_abb = [0, 0.5, 0, 0.5]
        self.two_gram_weights_one_half_abb_one_abaaa = [2, 1.5, 1, 0.5]

    def setup_patch(self):
        self.feature_space_builder_patch = patch('preimage.features.n_gram_feature_space.'
                                                 'build_feature_space_without_positions')

    def test_one_gram_one_y_normalized_feature_space_is_normalized(self):
        self.feature_space_builder_patch.start().return_value = self.feature_space_one_gram_abb

        feature_space = NGramFeatureSpace(self.alphabet, n=1, Y=self.abb, is_normalized=True)
        feature_space_normalized = numpy.array(feature_space._feature_space.todense())

        numpy.testing.assert_array_equal(feature_space_normalized, self.feature_space_normalized_one_gram_abb)

    def test_two_gram_one_y_normalized_feature_space_is_normalized(self):
        self.feature_space_builder_patch.start().return_value = self.feature_space_two_gram_abb

        feature_space = NGramFeatureSpace(self.alphabet, n=2, Y=self.abb, is_normalized=True)
        feature_space_normalized = numpy.array(feature_space._feature_space.todense())

        numpy.testing.assert_array_equal(feature_space_normalized, self.feature_space_normalized_two_gram_abb)

    def test_two_gram_two_y_normalized_feature_space_is_normalized(self):
        self.feature_space_builder_patch.start().return_value = self.feature_space_one_gram_abb_abaaa

        feature_space = NGramFeatureSpace(self.alphabet, n=1, Y=self.abb, is_normalized=True)
        feature_space_normalized = numpy.array(feature_space._feature_space.todense())

        numpy.testing.assert_array_equal(feature_space_normalized, self.feature_space_normalized_one_gram_abb_abaaa)

    def test_one_y_with_one_half_weight_compute_one_gram_weights_returns_expected_weights(self):
        self.feature_space_builder_patch.start().return_value = self.feature_space_one_gram_abb
        feature_space = NGramFeatureSpace(self.alphabet, n=1, Y=self.abb, is_normalized=False)

        n_gram_weights = feature_space.compute_weights(y_weights=numpy.array([0.5]))

        numpy.testing.assert_array_equal(n_gram_weights, self.one_gram_weights_one_half_abb)

    def test_one_y_with_one_half_weight_compute_two_gram_weights_returns_expected_weights(self):
        self.feature_space_builder_patch.start().return_value = self.feature_space_two_gram_abb
        feature_space = NGramFeatureSpace(self.alphabet, n=2, Y=self.abb, is_normalized=False)

        n_gram_weights = feature_space.compute_weights(y_weights=numpy.array([0.5]))

        numpy.testing.assert_array_equal(n_gram_weights, self.two_gram_weights_one_half_abb)

    def test_two_y_with_different_weights_compute_two_gram_weights_returns_expected_weights(self):
        self.feature_space_builder_patch.start().return_value = self.feature_space_two_gram_abb_abaaa
        feature_space = NGramFeatureSpace(self.alphabet, n=2, Y=self.abb_abaaa, is_normalized=False)

        n_gram_weights = feature_space.compute_weights(y_weights=numpy.array([0.5, 1]))

        numpy.testing.assert_array_equal(n_gram_weights, self.two_gram_weights_one_half_abb_one_abaaa)

    def test_compute_n_gram_weights_does_not_change_feature_space(self):
        self.feature_space_builder_patch.start().return_value = self.feature_space_one_gram_abb
        feature_space = NGramFeatureSpace(self.alphabet, n=1, Y=self.abb, is_normalized=False)

        feature_space.compute_weights(y_weights=numpy.array([0.5]))
        n_gram_feature_space = feature_space._feature_space

        numpy.testing.assert_array_equal(n_gram_feature_space.toarray(), self.feature_space_one_gram_abb.toarray())


if __name__ == '__main__':
    unittest2.main()