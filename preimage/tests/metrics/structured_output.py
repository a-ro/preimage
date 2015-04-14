__author__ = 'amelie'

import unittest2

from preimage.metrics.structured_output import zero_one_loss


class TestStructuredOutputMetrics(unittest2.TestCase):
    def setUp(self):
        self.unicorn = ['unicorn']
        self.unicor = ['unicor']
        self.potato = ['potato']
        self.tomato = ['tomato']
        self.unicorn_potato = self.unicorn + self.potato
        self.unicor_tomato = self.unicor + self.tomato
        self.unicor_potato = self.unicor + self.potato

    def test_same_word_zero_one_loss_is_zero(self):
        loss = zero_one_loss(self.unicorn, self.unicorn)
        self.assertEqual(loss, 0.)

    def test_same_words_zero_one_loss_is_zero(self):
        loss = zero_one_loss(self.unicorn_potato, self.unicorn_potato)
        self.assertEqual(loss, 0.)

    def test_not_same_word_zero_one_loss_is_one(self):
        loss = zero_one_loss(self.unicorn, self.unicor)
        self.assertEqual(loss, 1.)

    def test_not_same_words_zero_one_loss_is_one(self):
        loss = zero_one_loss(self.unicorn_potato, self.unicor_tomato)
        self.assertEqual(loss, 1.)

    def test_one_same_word_on_two_zero_one_loss_is_one_half(self):
        loss = zero_one_loss(self.unicorn_potato, self.unicor_potato)
        self.assertEqual(loss, 0.5)

    def test_not_same_word_count_zero_one_loss_throws_value_error(self):
        with self.assertRaises(ValueError):
            zero_one_loss(self.unicorn_potato, self.unicorn)


if __name__ == '__main__':
    unittest2.main()