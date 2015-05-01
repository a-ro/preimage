__author__ = 'amelie'

import unittest2
import numpy.testing

from preimage.utils import alphabet
from preimage.exceptions.n_gram import InvalidNGramLengthError


class TestAlphabet(unittest2.TestCase):
    def setUp(self):
        self.a_b_alphabet = ['a', 'b']
        self.two_grams = ['aa', 'ab', 'ba', 'bb']
        self.one_gram_to_index = {'a': 0, 'b': 1}
        self.two_gram_to_index = {'aa': 0, 'ab': 1, 'ba': 2, 'bb': 3}
        self.index_to_one_gram = {0: 'a', 1: 'b'}
        self.index_to_two_gram = {0: 'aa', 1: 'ab', 2: 'ba', 3: 'bb'}
        self.cab = ['cab']
        self.cab_aa = ['cab', 'aa']
        self.cab_int = [[99, 97, 98]]
        self.cab_aa_int = [[99, 97, 98], [97, 97, -1]]

    def test_integer_n_is_zero_get_n_grams_raises_value_error(self):
        with self.assertRaises(ValueError):
            alphabet.get_n_grams(self.a_b_alphabet, n=0.5)

    def test_get_one_grams_returns_alphabet(self):
        n_grams = alphabet.get_n_grams(self.a_b_alphabet, n=1)

        numpy.testing.assert_array_equal(n_grams, self.a_b_alphabet)

    def test_get_two_grams_returns_expected_two_grams(self):
        n_grams = alphabet.get_n_grams(self.a_b_alphabet, n=2)

        numpy.testing.assert_array_equal(n_grams, self.two_grams)

    def test_get_one_gram_to_index_returns_expected_dict(self):
        n_gram_to_index = alphabet.get_n_gram_to_index(self.a_b_alphabet, n=1)

        self.assertDictEqual(n_gram_to_index, self.one_gram_to_index)

    def test_get_two_gram_to_index_returns_expected_dict(self):
        n_gram_to_index = alphabet.get_n_gram_to_index(self.a_b_alphabet, n=2)

        self.assertDictEqual(n_gram_to_index, self.two_gram_to_index)

    def test_n_zero_get_n_gram_to_index_raises_value_error(self):
        with self.assertRaises(InvalidNGramLengthError):
            alphabet.get_n_gram_to_index(self.a_b_alphabet, n=0)

    def test_get_index_to_one_gram_returns_expected_dict(self):
        index_to_n_gram = alphabet.get_index_to_n_gram(self.a_b_alphabet, n=1)

        self.assertDictEqual(index_to_n_gram, self.index_to_one_gram)

    def test_get_index_to_two_gram_returns_expected_dict(self):
        index_to_n_gram = alphabet.get_index_to_n_gram(self.a_b_alphabet, n=1)

        self.assertDictEqual(index_to_n_gram, self.index_to_one_gram)

    def test_n_zero_get_index_to_n_gram_raises_value_error(self):
        with self.assertRaises(InvalidNGramLengthError):
            alphabet.get_index_to_n_gram(self.a_b_alphabet, n=0)

    def test_one_string_transform_strings_to_integer_returns_integer_string(self):
        Y_int = alphabet.transform_strings_to_integer_lists(Y=self.cab)

        numpy.testing.assert_array_equal(Y_int, self.cab_int)

    def test_two_strings_different_length_transform_strings_to_integer_returns_integer_strings(self):
        Y_int = alphabet.transform_strings_to_integer_lists(Y=self.cab_aa)

        numpy.testing.assert_array_equal(Y_int, self.cab_aa_int)


if __name__ == '__main__':
    unittest2.main()