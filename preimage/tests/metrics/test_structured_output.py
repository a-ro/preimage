__author__ = 'amelie'

import unittest2

from preimage.metrics.structured_output import zero_one_loss, hamming_loss, levenshtein_loss


class TestStructuredOutputMetrics(unittest2.TestCase):
    def setUp(self):
        self.unicorn = ['unicorn']
        self.unicarn = ['unicarn']
        self.nicornu = ['nicornu']
        self.unicor = ['unicor']
        self.potato = ['potato']
        self.tomato = ['tomato']
        self.a = ['a']
        self.unicorn_potato = self.unicorn + self.potato
        self.unicor_tomato = self.unicor + self.tomato
        self.unicor_potato = self.unicor + self.potato
        self.unicarn_tomato = self.unicarn + self.tomato
        self.nicornu_tomato = self.nicornu + self.tomato

    def test_same_word_zero_one_loss_is_zero(self):
        loss = zero_one_loss(self.unicorn, self.unicorn)

        self.assertEqual(loss, 0.)

    def test_not_same_word_zero_one_loss_is_one(self):
        loss = zero_one_loss(self.unicorn, self.unicor)

        self.assertEqual(loss, 1.)

    def test_one_same_word_on_two_zero_one_loss_is_one_half(self):
        loss = zero_one_loss(self.unicorn_potato, self.unicor_potato)

        self.assertEqual(loss, 0.5)

    def test_not_same_word_count_zero_one_loss_throws_value_error(self):
        with self.assertRaises(ValueError):
            zero_one_loss(self.unicorn_potato, self.unicorn)

    def test_same_word_hamming_loss_is_zero(self):
        loss = hamming_loss(self.unicorn, self.unicorn)

        self.assertEqual(loss, 0.)

    def test_one_wrong_letter_on_seven_hamming_loss_is_one_on_seven(self):
        loss = hamming_loss(self.unicorn, self.unicarn)

        self.assertEqual(loss, 1./7)

    def test_all_wrong_letters_hamming_loss_is_one(self):
        loss = hamming_loss(self.unicorn, self.nicornu)

        self.assertEqual(loss, 1.)

    def test_two_words_with_errors_hamming_loss_is_mean_of_errors(self):
        loss = hamming_loss(self.unicorn_potato, self.unicarn_tomato)
        mean_errors = (1./7 + 2./6) / 2

        self.assertEqual(loss, mean_errors)

    def test_not_same_word_count_hamming_loss_throws_value_error(self):
        with self.assertRaises(ValueError):
            hamming_loss(self.unicorn_potato, self.unicorn)

    def test_not_same_letter_count_hamming_loss_throws_value_error(self):
        with self.assertRaises(ValueError):
            hamming_loss(self.unicorn, self.unicor)

    def test_same_word_levenshtein_loss_is_zero(self):
        loss = levenshtein_loss(self.unicorn, self.unicorn)

        self.assertEqual(loss, 0.)

    def test_one_missing_letter_on_seven_levenshtein_loss_is_one_on_seven(self):
        loss = levenshtein_loss(self.unicorn, self.unicor)

        self.assertEqual(loss, 1./7)

    def test_one_extra_letter_on_seven_levenshtein_loss_is_one_on_seven(self):
        loss = levenshtein_loss(self.unicor, self.unicorn)

        self.assertEqual(loss, 1./7)

    def test_no_letters_in_common_levenshtein_loss_is_one(self):
        loss = levenshtein_loss(self.unicorn, self.a)

        self.assertEqual(loss, 1.)

    def test_two_words_with_errors_levenshtein_loss_is_mean_of_errors(self):
        loss = levenshtein_loss(self.unicorn_potato, self.nicornu_tomato)
        mean_errors = (2./7 + 2./6) / 2

        self.assertEqual(loss, mean_errors)

    def test_not_same_word_count_levenshtein_loss_throws_value_error(self):
        with self.assertRaises(ValueError):
            levenshtein_loss(self.unicorn_potato, self.unicorn)


if __name__ == '__main__':
    unittest2.main()