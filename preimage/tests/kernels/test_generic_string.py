__author__ = 'amelie'

from math import sqrt

import unittest2
import numpy.testing
from mock import patch

from preimage.kernels.generic_string import GenericStringKernel, element_wise_kernel


class TestGenericStringKernel(unittest2.TestCase):
    def setUp(self):
        self.setup_alphabet()
        self.setup_positions()
        self.setup_loader()
        self.setup_similarity()
        self.setup_sigma_c_similarity()

    def setup_alphabet(self):
        self.alphabet = ['a', 'b']
        self.aba = ['aba']
        self.bbbb = ['bbbb']
        self.aba_bbbb = ['aba', 'bbbb']
        self.aba_int = numpy.array([[0, 1, 0]], dtype=numpy.int8)
        self.bbbb_int = numpy.array([[1, 1, 1, 1]], dtype=numpy.int8)
        self.aba_bbbb_int = numpy.array([[0, 1, 0, -1], [1, 1, 1, 1]],
                                        dtype=numpy.int8)
        self.string_to_int_patch = patch('preimage.kernels.generic_string.transform_strings_to_integer_lists')

    def setup_positions(self):
        self.small_sigma = 1e-8
        self.medium_sigma = 1
        self.large_sigma = 1e8
        self.positions_small_sigma = numpy.eye(4)
        self.positions_medium_sigma = numpy.array([[1, 0.5, 0, 0], [0.5, 1, 0.5, 0], [0, 0.5, 1, 0.5], [0, 0, 0.5, 1]])
        self.positions_large_sigma = numpy.ones((4, 4))
        self.position_patch = patch('preimage.kernels.generic_string.compute_position_weights_matrix')

    def setup_loader(self):
        self.loader_patch = patch('preimage.kernels.generic_string.load_amino_acids_and_descriptors')
        self.loader_patch.start().return_value = (['a', 'b'], [[1., 3], [3, 2]])

    def setup_similarity(self):
        self.aba_small_sigma_similarity = [3]
        self.bbbb_medium_sigma_similarity = [7]
        self.aba_large_sigma_similarity = [5]
        self.aba_large_sigma_p_sigma_c_similarity = [9]
        self.bbbb_two_gram_large_sigma_similarity = [9]
        self.aba_bbbb_small_sigma_similarity = [3, 4]

    def setup_sigma_c_similarity(self):
        normalized_descriptors = numpy.array([[1. / sqrt(10), 3. / sqrt(10)], [3. / sqrt(13), 2. / sqrt(13)]])
        a_b_distance = (normalized_descriptors[0, 0] - normalized_descriptors[1, 0]) ** 2 + \
                       (normalized_descriptors[0, 1] - normalized_descriptors[1, 1]) ** 2
        a_b_sigma_c_similarity = numpy.exp(-a_b_distance / 2)
        ab_ba_two_gram_medium_sigma_c_similarity = numpy.exp(-a_b_distance / 2) ** 2
        self.aba_medium_sigma_c_similarity = 5 + 4 * a_b_sigma_c_similarity
        self.aba_two_gram_medium_sigma_c_similarity = self.aba_medium_sigma_c_similarity + 2 + \
                                                      2 * ab_ba_two_gram_medium_sigma_c_similarity
        self.aba_bbbb_medium_sigma_c_small_sigma_p = 1 + 2 * a_b_sigma_c_similarity
        self.aba_bbbb_normalized_small_sigma_similarity = 1. / sqrt(3 * 4)
        self.aba_bbbb_gram_matrix_normalized = [[1., self.aba_bbbb_normalized_small_sigma_similarity],
                                                [self.aba_bbbb_normalized_small_sigma_similarity, 1.]]


    def test_one_gram_one_x_small_sigma_element_wise_gs_kernel_returns_expected_value(self):
        self.position_patch.start().return_value = self.positions_small_sigma
        self.string_to_int_patch.start().return_value = self.aba_int

        similarities = element_wise_kernel(self.aba, self.small_sigma, n=1, alphabet=self.alphabet)

        numpy.testing.assert_array_equal(similarities, self.aba_small_sigma_similarity)

    def test_one_gram_two_x_small_sigma_element_wise_gs_kernel_returns_expected_values(self):
        self.position_patch.start().return_value = self.positions_small_sigma
        self.string_to_int_patch.start().return_value = self.aba_bbbb_int

        similarities = element_wise_kernel(self.aba_bbbb, self.small_sigma, n=1, alphabet=self.alphabet)

        numpy.testing.assert_array_equal(similarities, self.aba_bbbb_small_sigma_similarity)

    def test_one_gram_one_x_large_sigma_element_wise_gs_kernel_returns_expected_value(self):
        self.position_patch.start().return_value = self.positions_large_sigma
        self.string_to_int_patch.start().return_value = self.aba_int

        similarities = element_wise_kernel(self.aba, self.large_sigma, n=1, alphabet=self.alphabet)

        numpy.testing.assert_array_equal(similarities, self.aba_large_sigma_similarity)

    def test_one_gram_one_x_medium_sigma_element_wise_gs_kernel_returns_expected_value(self):
        self.position_patch.start().return_value = self.positions_medium_sigma
        self.string_to_int_patch.start().return_value = self.bbbb_int

        similarities = element_wise_kernel(self.bbbb, self.medium_sigma, n=1, alphabet=self.alphabet)

        numpy.testing.assert_array_equal(similarities, self.bbbb_medium_sigma_similarity)

    def test_two_gram_one_x_small_sigma_element_wise_gs_kernel_returns_expected_value(self):
        self.position_patch.start().return_value = self.positions_large_sigma
        self.string_to_int_patch.start().return_value = self.bbbb_int

        similarities = element_wise_kernel(self.bbbb, self.large_sigma, n=2, alphabet=self.alphabet)

        numpy.testing.assert_array_equal(similarities, self.bbbb_two_gram_large_sigma_similarity)

    def test_same_string_normalized_gs_kernel_returns_one(self):
        self.position_patch.start().return_value = self.positions_small_sigma
        self.string_to_int_patch.start().side_effect = [self.aba_int, self.aba_int]
        kernel = GenericStringKernel()

        gram_matrix = kernel(self.aba, self.aba)

        numpy.testing.assert_array_equal(gram_matrix, [[1]])

    def test_same_string_not_normalized_small_sigma_position_gs_kernel_returns_expected_value(self):
        self.position_patch.start().return_value = self.positions_small_sigma
        self.string_to_int_patch.start().side_effect = [self.aba_int, self.aba_int]
        kernel = GenericStringKernel(is_normalized=False, sigma_position=self.small_sigma, n=1)

        gram_matrix = kernel(self.aba, self.aba)

        numpy.testing.assert_array_equal(gram_matrix, [self.aba_small_sigma_similarity])

    def test_same_string_not_normalized_large_sigma_p_small_sigma_c_gs_kernel_returns_expected_value(self):
        self.position_patch.start().return_value = self.positions_large_sigma
        self.string_to_int_patch.start().side_effect = [self.aba_int, self.aba_int]
        kernel = GenericStringKernel(is_normalized=False, sigma_position=self.large_sigma,
                                     sigma_amino_acid=self.small_sigma, n=1)

        gram_matrix = kernel(self.aba, self.aba)

        numpy.testing.assert_array_equal(gram_matrix, [self.aba_large_sigma_similarity])

    def test_same_string_not_normalized_large_sigma_p_large_sigma_c_gs_kernel_returns_expected_value(self):
        self.position_patch.start().return_value = self.positions_large_sigma
        self.string_to_int_patch.start().side_effect = [self.aba_int, self.aba_int]
        kernel = GenericStringKernel(is_normalized=False, sigma_position=self.large_sigma,
                                     sigma_amino_acid=self.large_sigma, n=1)

        gram_matrix = kernel(self.aba, self.aba)

        numpy.testing.assert_array_equal(gram_matrix, [self.aba_large_sigma_p_sigma_c_similarity])

    def test_same_string_large_sigma_p_medium_sigma_c_gs_kernel_returns_expected_value(self):
        self.position_patch.start().return_value = self.positions_large_sigma
        self.string_to_int_patch.start().side_effect = [self.aba_int, self.aba_int]
        kernel = GenericStringKernel(is_normalized=False, sigma_position=self.large_sigma,
                                     sigma_amino_acid=self.medium_sigma, n=1)

        gram_matrix = kernel(self.aba, self.aba)

        numpy.testing.assert_almost_equal(gram_matrix, [[self.aba_medium_sigma_c_similarity]])

    def test_two_gram_same_string_large_sigma_p_medium_sigma_c_gs_kernel_returns_expected_value(self):
        self.position_patch.start().return_value = self.positions_large_sigma
        self.string_to_int_patch.start().side_effect = [self.aba_int, self.aba_int]
        kernel = GenericStringKernel(is_normalized=False, sigma_position=self.large_sigma,
                                     sigma_amino_acid=self.medium_sigma, n=2)

        gram_matrix = kernel(self.aba, self.aba)

        numpy.testing.assert_almost_equal(gram_matrix, [[self.aba_two_gram_medium_sigma_c_similarity]])

    def test_one_gram_different_string_small_sigma_p_medium_sigma_c_gs_kernel_returns_expected_value(self):
        self.position_patch.start().return_value = self.positions_small_sigma
        self.string_to_int_patch.start().side_effect = [self.aba_int, self.bbbb_int]
        kernel = GenericStringKernel(is_normalized=False, sigma_position=self.small_sigma,
                                     sigma_amino_acid=self.medium_sigma, n=1)

        gram_matrix = kernel(self.aba, self.bbbb)

        numpy.testing.assert_almost_equal(gram_matrix, [[self.aba_bbbb_medium_sigma_c_small_sigma_p]])

    def test_one_gram_normalized_two_strings_small_sigmas_gs_kernel_returns_expected_value(self):
        self.position_patch.start().return_value = self.positions_small_sigma
        self.string_to_int_patch.start().side_effect = [self.aba_int, self.bbbb_int]
        kernel = GenericStringKernel(is_normalized=True, sigma_position=self.small_sigma,
                                     sigma_amino_acid=self.small_sigma, n=1)

        gram_matrix = kernel(self.aba, self.bbbb)

        numpy.testing.assert_almost_equal(gram_matrix, [[self.aba_bbbb_normalized_small_sigma_similarity]])

    def test_one_gram_symmetric_normalized_gs_kernel_returns_expected_value(self):
        self.position_patch.start().return_value = self.positions_small_sigma
        self.string_to_int_patch.start().side_effect = [self.aba_bbbb_int, self.aba_bbbb_int]
        kernel = GenericStringKernel(is_normalized=True, sigma_position=self.small_sigma,
                                     sigma_amino_acid=self.small_sigma, n=1)

        gram_matrix = kernel(self.aba_bbbb, self.aba_bbbb)

        numpy.testing.assert_almost_equal(gram_matrix, self.aba_bbbb_gram_matrix_normalized)

    def test_two_gram_same_string_large_sigma_p_medium_sigma_c_gs_element_wise_kernel_returns_expected_value(self):
        self.position_patch.start().return_value = self.positions_large_sigma
        self.string_to_int_patch.start().side_effect = [self.aba_int, self.aba_int]
        kernel = GenericStringKernel(sigma_position=self.large_sigma, sigma_amino_acid=self.medium_sigma, n=2)

        similarities = kernel.element_wise_kernel(self.aba)

        numpy.testing.assert_almost_equal(similarities, [self.aba_two_gram_medium_sigma_c_similarity])

    def test_one_gram_two_strings_small_sigmas_gs_kernel_returns_expected_value(self):
        self.position_patch.start().return_value = self.positions_small_sigma
        self.string_to_int_patch.start().return_value = self.aba_bbbb_int
        kernel = GenericStringKernel(sigma_position=self.small_sigma, sigma_amino_acid=self.small_sigma, n=1)

        gram_matrix = kernel.element_wise_kernel(self.aba_bbbb)

        numpy.testing.assert_almost_equal(gram_matrix, self.aba_bbbb_small_sigma_similarity)


if __name__ == '__main__':
    unittest2.main()