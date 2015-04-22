__author__ = 'amelie'

import unittest2
import numpy.testing

from preimage.utils.alphabet import get_n_gram_to_index, get_index_to_n_gram, get_n_grams
from preimage.exceptions.n_gram import InvalidNGramLengthError


class TestAlphabet(unittest2.TestCase):
    def setUp(self):
        self.a_b_alphabet = ['a', 'b']
        self.two_grams = ['aa', 'ab', 'ba', 'bb']
        self.one_gram_to_index = {'a': 0, 'b': 1}
        self.two_gram_to_index = {'aa': 0, 'ab': 1, 'ba': 2, 'bb': 3}
        self.index_to_one_gram = {0: 'a', 1: 'b'}
        self.index_to_two_gram = {0: 'aa', 1: 'ab', 2: 'ba', 3: 'bb'}

    def test_integer_n_is_zero_get_n_grams_raises_value_error(self):
        with self.assertRaises(ValueError):
            get_n_grams(self.a_b_alphabet, n=0.5)

    def test_get_one_grams_returns_alphabet(self):
        n_grams = get_n_grams(self.a_b_alphabet, n=1)

        numpy.testing.assert_array_equal(n_grams, self.a_b_alphabet)

    def test_get_two_grams_returns_expected_two_grams(self):
        n_grams = get_n_grams(self.a_b_alphabet, n=2)

        numpy.testing.assert_array_equal(n_grams, self.two_grams)

    def test_get_one_gram_to_index_returns_expected_dict(self):
        n_gram_to_index = get_n_gram_to_index(self.a_b_alphabet, n=1)

        self.assertDictEqual(n_gram_to_index, self.one_gram_to_index)

    def test_get_two_gram_to_index_returns_expected_dict(self):
        n_gram_to_index = get_n_gram_to_index(self.a_b_alphabet, n=2)

        self.assertDictEqual(n_gram_to_index, self.two_gram_to_index)

    def test_n_zero_get_n_gram_to_index_raises_value_error(self):
        with self.assertRaises(InvalidNGramLengthError):
            get_n_gram_to_index(self.a_b_alphabet, n=0)

    def test_get_index_to_one_gram_returns_expected_dict(self):
        index_to_n_gram = get_index_to_n_gram(self.a_b_alphabet, n=1)

        self.assertDictEqual(index_to_n_gram, self.index_to_one_gram)

    def test_get_index_to_two_gram_returns_expected_dict(self):
        index_to_n_gram = get_index_to_n_gram(self.a_b_alphabet, n=1)

        self.assertDictEqual(index_to_n_gram, self.index_to_one_gram)

    def test_n_zero_get_index_to_n_gram_raises_value_error(self):
        with self.assertRaises(InvalidNGramLengthError):
            get_index_to_n_gram(self.a_b_alphabet, n=0)


if __name__ == '__main__':
    unittest2.main()