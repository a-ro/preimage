__author__ = 'amelie'

import unittest2
import numpy.testing
from mock import patch

from preimage.inference.n_gram_feature_space import NGramFeatureSpace
from preimage.exceptions.n_gram import InvalidNGramError


class TestNGramFeatureSpace(unittest2.TestCase):
    def setUp(self):
        self.alphabet = ['a', 'b']
        self.one_gram_to_index = {'a': 0, 'b': 1}
        self.two_gram_to_index = {'aa': 0, 'ab': 1, 'ba': 2, 'bb': 3}
        self.two_gram_to_index_without_bb = {'aa': 0, 'ab': 1, 'ba': 2}
        self.b = ['b']
        self.abb = ['abb']
        self.abaaa = ['abaaa']
        self.abb_abaaa = self.abb + self.abaaa
        self.Feature_space_one_gram_b = [[0, 1]]
        self.Feature_space_one_gram_abb = [[1, 2]]
        self.Feature_space_two_gram_abaaa = [[2, 1, 1, 0]]
        self.Feature_space_two_gram_abb_abaaa = [[0, 1, 0, 1], [2, 1, 1, 0]]
        self.n_gram_to_index_patch = patch('preimage.inference.n_gram_feature_space.get_n_gram_to_index')
        self.one_gram_weights_one_half_abb = [0.5, 1]
        self.two_gram_weights_one_half_abb = [0, 0.5, 0, 0.5]
        self.two_gram_weights_one_half_abb_one_abaaa = [2, 1.5, 1, 0.5]

    def test_one_gram_one_letter_y_n_gram_feature_space_has_one_n_gram(self):
        self.n_gram_to_index_patch.start().return_value = self.one_gram_to_index

        feature_space = NGramFeatureSpace(n=1, alphabet=self.alphabet, Y=self.b)
        dense_feature_space = feature_space._N_gram_feature_space.toarray()

        numpy.testing.assert_array_equal(dense_feature_space, self.Feature_space_one_gram_b)

    def test_one_gram_three_letter_y_n_gram_feature_space_has_three_n_grams(self):
        self.n_gram_to_index_patch.start().return_value = self.one_gram_to_index

        feature_space = NGramFeatureSpace(n=1, alphabet=self.alphabet, Y=self.abb)
        dense_feature_space = feature_space._N_gram_feature_space.toarray()

        numpy.testing.assert_array_equal(dense_feature_space, self.Feature_space_one_gram_abb)

    def test_two_gram_five_letter_y_n_gram_feature_space_has_four_two_grams(self):
        self.n_gram_to_index_patch.start().return_value = self.two_gram_to_index

        feature_space = NGramFeatureSpace(n=2, alphabet=self.alphabet, Y=self.abaaa)
        dense_feature_space = feature_space._N_gram_feature_space.toarray()

        numpy.testing.assert_array_equal(dense_feature_space, self.Feature_space_two_gram_abaaa)

    def test_two_gram_two_y_n_gram_feature_space_builds_expected_feature_space(self):
        self.n_gram_to_index_patch.start().return_value = self.two_gram_to_index

        feature_space = NGramFeatureSpace(n=2, alphabet=self.alphabet, Y=self.abb_abaaa)
        dense_feature_space = feature_space._N_gram_feature_space.toarray()

        numpy.testing.assert_array_equal(dense_feature_space, self.Feature_space_two_gram_abb_abaaa)

    def test_two_gram_not_in_alphabet_n_gram_feature_space_raises_error(self):
        self.n_gram_to_index_patch.start().return_value = self.two_gram_to_index_without_bb

        with self.assertRaises(InvalidNGramError):
            NGramFeatureSpace(n=2, alphabet=self.alphabet, Y=self.abb)

    def test_one_y_with_one_half_weight_compute_one_gram_weights_returns_expected_weights(self):
        self.n_gram_to_index_patch.start().return_value = self.one_gram_to_index

        feature_space = NGramFeatureSpace(n=1, alphabet=self.alphabet, Y=self.abb)
        n_gram_weights = feature_space.compute_weights(Y_weights=numpy.array([[0.5]]))

        numpy.testing.assert_array_equal(n_gram_weights, self.one_gram_weights_one_half_abb)

    def test_one_y_with_one_half_weight_compute_two_gram_weights_returns_expected_weights(self):
        self.n_gram_to_index_patch.start().return_value = self.two_gram_to_index

        feature_space = NGramFeatureSpace(n=2, alphabet=self.alphabet, Y=self.abb)
        n_gram_weights = feature_space.compute_weights(Y_weights=numpy.array([[0.5]]))

        numpy.testing.assert_array_equal(n_gram_weights, self.two_gram_weights_one_half_abb)

    def test_two_y_with_different_weights_compute_two_gram_weights_returns_expected_weights(self):
        self.n_gram_to_index_patch.start().return_value = self.two_gram_to_index

        feature_space = NGramFeatureSpace(n=2, alphabet=self.alphabet, Y=self.abb_abaaa)
        n_gram_weights = feature_space.compute_weights(Y_weights=numpy.array([[0.5, 1]]))

        numpy.testing.assert_array_equal(n_gram_weights, self.two_gram_weights_one_half_abb_one_abaaa)

    def test_compute_n_gram_weights_does_not_change_feature_space(self):
        self.n_gram_to_index_patch.start().return_value = self.one_gram_to_index
        feature_space = NGramFeatureSpace(n=1, alphabet=self.alphabet, Y=self.abb)

        feature_space.compute_weights(Y_weights=numpy.array([[0.5]]))
        n_gram_feature_space = feature_space._N_gram_feature_space.toarray()

        numpy.testing.assert_array_equal(n_gram_feature_space, self.Feature_space_one_gram_abb)


if __name__ == '__main__':
    unittest2.main()