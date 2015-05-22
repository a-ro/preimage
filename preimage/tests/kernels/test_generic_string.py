__author__ = 'amelie'

import unittest2
import numpy.testing
from mock import patch

from preimage.kernels.generic_string import element_wise_kernel


class TestGenericStringKernel(unittest2.TestCase):
    def setUp(self):
        self.setup_alphabet()
        self.setup_positions()
        self.setup_similarity()

    def setup_alphabet(self):
        self.aba = ['aba']
        self.bbbb = ['bbbb']
        self.aba_bbbb = ['aba', 'bbbb']
        self.aba_int = numpy.array([[0, 1, 0]], dtype=numpy.int8)
        self.bbbb_int = numpy.array([[1, 1, 1, 1]], dtype=numpy.int8)
        self.aba_bbbb_int = numpy.array([[0, 1, 0, -1], [1, 1, 1, 1]], dtype=numpy.int8)
        self.string_to_int_patch = patch('preimage.kernels.generic_string.transform_strings_to_integer_lists')

    def setup_positions(self):
        self.small_sigma = 1e-8
        self.medium_sigma = 1
        self.large_sigma = 1e8
        self.positions_small_sigma = numpy.eye(4)
        self.positions_medium_sigma = numpy.array([[1, 0.5, 0, 0], [0.5, 1, 0.5, 0], [0, 0.5, 1, 0.5], [0, 0, 0.5, 1]])
        self.positions_large_sigma = numpy.ones((4, 4))
        self.position_patch = patch('preimage.kernels.generic_string.compute_position_weights_matrix')

    def setup_similarity(self):
        self.aba_small_sigma_similarity = [3]
        self.bbbb_medium_sigma_similarity = [7]
        self.aba_large_sigma_similarity = [5]
        self.bbbb_two_gram_large_sigma_similarity = [9]
        self.aba_bbbb_small_sigma_similarity = [3, 4]

    def test_one_gram_one_x_small_sigma_element_wise_gs_kernel_returns_expected_value(self):
        self.position_patch.start().return_value = self.positions_small_sigma
        self.string_to_int_patch.start().return_value = self.aba_int

        similarities = element_wise_kernel(self.aba, self.small_sigma, n=1)

        numpy.testing.assert_array_equal(similarities, self.aba_small_sigma_similarity)

    def test_one_gram_two_x_small_sigma_element_wise_gs_kernel_returns_expected_values(self):
        self.position_patch.start().return_value = self.positions_small_sigma
        self.string_to_int_patch.start().return_value = self.aba_bbbb_int

        similarities = element_wise_kernel(self.aba_bbbb, self.small_sigma, n=1)

        numpy.testing.assert_array_equal(similarities, self.aba_bbbb_small_sigma_similarity)

    def test_one_gram_one_x_large_sigma_element_wise_gs_kernel_returns_expected_value(self):
        self.position_patch.start().return_value = self.positions_large_sigma
        self.string_to_int_patch.start().return_value = self.aba_int

        similarities = element_wise_kernel(self.aba, self.large_sigma, n=1)

        numpy.testing.assert_array_equal(similarities, self.aba_large_sigma_similarity)

    def test_one_gram_one_x_medium_sigma_element_wise_gs_kernel_returns_expected_value(self):
        self.position_patch.start().return_value = self.positions_medium_sigma
        self.string_to_int_patch.start().return_value = self.bbbb_int

        similarities = element_wise_kernel(self.bbbb, self.medium_sigma, n=1)

        numpy.testing.assert_array_equal(similarities, self.bbbb_medium_sigma_similarity)

    def test_two_gram_one_x_small_sigma_element_wise_gs_kernel_returns_expected_value(self):
        self.position_patch.start().return_value = self.positions_large_sigma
        self.string_to_int_patch.start().return_value = self.bbbb_int

        similarities = element_wise_kernel(self.bbbb, self.large_sigma, n=2)

        numpy.testing.assert_array_equal(similarities, self.bbbb_two_gram_large_sigma_similarity)


if __name__ == '__main__':
    unittest2.main()