__author__ = 'amelie'

import unittest2
import numpy.testing
from mock import patch, MagicMock

from preimage.inference.n_gram_feature_space import NGramFeatureSpace


# Todo add test on bad constructor value : n <=0
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
        self.b_one_gram_feature_space = [[0, 1]]
        self.abb_one_gram_feature_space = [[1, 2]]
        self.abaaa_two_gram_feature_space = [[2, 1, 1, 0]]
        self.abb_abaaa_two_gram_feature_space = [[0, 1, 0, 1], [2, 1, 1, 0]]
        self.n_gram_to_index_patch = patch('preimage.inference.n_gram_feature_space.get_n_gram_to_index')

    def test_one_gram_one_letter_example_n_gram_feature_space_has_one_n_gram(self):
        self.n_gram_to_index_patch.start().return_value = self.one_gram_to_index

        feature_space = NGramFeatureSpace(n=1, alphabet=self.alphabet, Y=self.b)
        dense_feature_space = feature_space._N_gram_feature_space.toarray()

        numpy.testing.assert_array_equal(dense_feature_space, self.b_one_gram_feature_space)

    def test_one_gram_three_letter_example_n_gram_feature_space_has_three_n_grams(self):
        self.n_gram_to_index_patch.start().return_value = self.one_gram_to_index

        feature_space = NGramFeatureSpace(n=1, alphabet=self.alphabet, Y=self.abb)
        dense_feature_space = feature_space._N_gram_feature_space.toarray()

        numpy.testing.assert_array_equal(dense_feature_space, self.abb_one_gram_feature_space)

    def test_two_gram_five_letter_example_n_gram_feature_space_has_four_two_grams(self):
        self.n_gram_to_index_patch.start().return_value = self.two_gram_to_index

        feature_space = NGramFeatureSpace(n=2, alphabet=self.alphabet, Y=self.abaaa)
        dense_feature_space = feature_space._N_gram_feature_space.toarray()

        numpy.testing.assert_array_equal(dense_feature_space, self.abaaa_two_gram_feature_space)

    def test_two_gram_two_examples_n_gram_feature_space_builds_expected_feature_space(self):
        self.n_gram_to_index_patch.start().return_value = self.two_gram_to_index

        feature_space = NGramFeatureSpace(n=2, alphabet=self.alphabet, Y=self.abb_abaaa)
        dense_feature_space = feature_space._N_gram_feature_space.toarray()

        numpy.testing.assert_array_equal(dense_feature_space, self.abb_abaaa_two_gram_feature_space)

    def test_two_gram_not_in_alphabet_n_gram_feature_space_raises_value_error(self):
        self.n_gram_to_index_patch.start().return_value = self.two_gram_to_index_without_bb

        with self.assertRaises(ValueError):
            NGramFeatureSpace(n=2, alphabet=self.alphabet, Y=self.abb)


if __name__ == '__main__':
    unittest2.main()