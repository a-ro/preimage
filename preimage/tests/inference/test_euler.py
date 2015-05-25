__author__ = 'amelie'

import unittest2
import numpy.testing
from mock import patch

from preimage.inference.euler import EulerianPath
from preimage.exceptions.n_gram import InvalidNGramLengthError, InvalidYLengthError, NoThresholdsError
from preimage.exceptions.shape import InvalidShapeError


class TestEulerianPath(unittest2.TestCase):
    def setUp(self):
        numpy.random.seed(42)
        self.setup_alphabet()
        self.setup_weights()
        self.setup_thresholds()
        self.eulerian_path_algorithm = EulerianPath(self.alphabet, n=2, min_length=2, is_merging_path=False)

    def setup_alphabet(self):
        self.alphabet = ['a', 'b', 'c']
        self.small_alphabet = ['a', 'b']
        self.aabc_aacb = ['aabc', 'aacb']
        self.cac_aca = ['cac', 'aca']
        self.abba_bbab_babb = ['abba', 'bbab', 'babb']
        self.ccbc_cbcc_bccb = ['ccbc', 'cbcc', 'bccb']
        self.ccbcaa_cbccaa_bccbaa = ['ccbcaa', 'cbccaa', 'bccbaa']
        self.aaaabb_bbbbaa = ['aaaabb', 'bbbbaa']
        self.index_to_n_gram_patch = patch('preimage.inference.euler.get_index_to_n_gram')
        self.index_to_two_gram = {0: 'aa', 1: 'ab', 2: 'ac', 3: 'ba', 4: 'bb', 5: 'bc', 6: 'ca', 7: 'cb', 8: 'cc'}
        self.index_to_three_gram = {0: 'aaa', 1: 'aab', 2: 'aba', 3: 'abb', 4: 'baa', 5: 'bab', 6: 'bba', 7: 'bbb'}
        self.index_to_n_gram_patch.start().return_value = self.index_to_two_gram

    def setup_weights(self):
        self.a_b_c_weights = numpy.array([1., 1., 1.])
        self.ab_weights = numpy.array([0, 1., 0, 0, 0, 0, 0, 0, 0])
        self.ab_ca_weights = numpy.array([0, 1., 0, 0, 0, 0, 1., 0, 0])
        self.ab_ba_ca_weights = numpy.array([0, 1., 0, 1., 0, 0, 1., 0, 0])
        self.ab_bb_cc_weights = numpy.array([0, 1., 0, 0, 1., 0, 0, 0, 1.])
        self.ab_ba_bb_weights = numpy.array([0, 1., 0, 1., 1., 0, 0, 0, 0])
        self.aa_aa_bc_cc_cb_weights = numpy.array([2., 0, 0, 0., 0, 1., 0, 1., 1.])
        self.aa_ab_ac_weights = numpy.array([1., 1., 1., 0., 0, 0, 0, 0, 0])
        self.aab_weights = numpy.array([0, 1., 0, 0., 0, 0, 0, 0])
        self.aab_aab_weights = numpy.array([0, 2., 0, 0., 0, 0, 0, 0])
        self.aaa_bbb_weights = numpy.array([2., 0., 0, 0., 0, 0, 0, 2.])
        self.bc_cc_non_rounded_weights = numpy.array([0, 0, 0, 0, 0, 0.6, 0, 0, 1.5])
        self.bb_bc_non_rounded_weights = numpy.array([0, 0, 0, 0, 0.4, 0.6, 0, 0, 0])
        self.bb_bb_non_rounded_weights = numpy.array([0, 0, 0, 0, 0.6, 0.1, 0, 0, 0])

    def setup_thresholds(self):
        self.thresholds = numpy.arange(1, 10)
        self.high_tresholds = numpy.arange(5, 14)
        self.wrong_thresholds = numpy.array([1, 2, 3])
        self.ac_threshold_weights = numpy.array([0, 1., 4., 2., 0, 0, 3., 1., 0])
        self.aa_ac_threshold_weights = numpy.array([1.5, 1., 4., 2., 0, 0, 1., 1., 0])

    def test_one_n_gram_find_eulerian_path_returns_n_gram(self):
        y = self.eulerian_path_algorithm.find_eulerian_path(self.ab_weights, y_length=2)

        self.assertEqual(y, 'ab')

    def test_two_n_gram_find_eulerian_path_returns_two_n_gram_path(self):
        y = self.eulerian_path_algorithm.find_eulerian_path(self.ab_ca_weights, y_length=3)

        self.assertEqual(y, 'cab')

    def test_three_n_gram_find_eulerian_path_returns_three_n_gram_path(self):
        y = self.eulerian_path_algorithm.find_eulerian_path(self.ab_ba_ca_weights, y_length=4)

        self.assertEqual(y, 'caba')

    def test_three_n_gram_one_not_connected_find_eulerian_path_returns_longest_path(self):
        y = self.eulerian_path_algorithm.find_eulerian_path(self.ab_bb_cc_weights, y_length=4)

        self.assertEqual(y, 'abb')

    def test_merge_path_three_n_gram_one_not_connected_find_eulerian_path_returns_merged_path(self):
        self.eulerian_path_algorithm = EulerianPath(self.alphabet, n=2, min_length=2, is_merging_path=True)

        y = self.eulerian_path_algorithm.find_eulerian_path(self.ab_bb_cc_weights, y_length=4)

        self.assertEqual(y, 'abbc')

    def test_merge_path_three_connected_n_gram_find_eulerian_path_returns_longest_path(self):
        self.eulerian_path_algorithm = EulerianPath(self.alphabet, n=2, min_length=2, is_merging_path=True)

        y = self.eulerian_path_algorithm.find_eulerian_path(self.ab_ba_ca_weights, y_length=4)

        self.assertEqual(y, 'caba')

    def test_eulerian_cycle_three_n_gram_find_eulerian_path_returns_eulerian_cycle(self):
        y = self.eulerian_path_algorithm.find_eulerian_path(self.ab_ba_bb_weights, y_length=4)

        self.assertIn(y, self.abba_bbab_babb)

    def test_merge_path_eulerian_cycle_n_gram_find_eulerian_path_returns_eulerian_cycle(self):
        self.eulerian_path_algorithm = EulerianPath(self.alphabet, n=2, min_length=2, is_merging_path=True)

        y = self.eulerian_path_algorithm.find_eulerian_path(self.ab_ba_bb_weights, y_length=4)

        self.assertIn(y, self.abba_bbab_babb)

    def test_two_eulerian_cycles_find_eulerian_path_returns_longest_eulerian_cycle(self):
        y = self.eulerian_path_algorithm.find_eulerian_path(self.aa_aa_bc_cc_cb_weights, y_length=6)

        self.assertIn(y, self.ccbc_cbcc_bccb)

    def test_merge_path_two_eulerian_cycles_find_eulerian_path_returns_merged_eulerian_cycles(self):
        self.eulerian_path_algorithm = EulerianPath(self.alphabet, n=2, min_length=2, is_merging_path=True)

        y = self.eulerian_path_algorithm.find_eulerian_path(self.aa_aa_bc_cc_cb_weights, y_length=6)

        self.assertIn(y, self.ccbcaa_cbccaa_bccbaa)

    def test_three_n_gram_one_with_two_edges_find_eulerian_path_returns_path(self):
        y = self.eulerian_path_algorithm.find_eulerian_path(self.aa_ab_ac_weights, y_length=4)

        self.assertIn(y, self.aabc_aacb)

    def test_one_three_gram_find_eulerian_path_returns_three_gram(self):
        self.index_to_n_gram_patch.start().return_value = self.index_to_three_gram
        eulerian_path_algorithm = EulerianPath(self.small_alphabet, n=3, min_length=3, is_merging_path=False)

        y = eulerian_path_algorithm.find_eulerian_path(self.aab_weights, y_length=3)

        self.assertEqual(y, 'aab')

    def test_two_three_gram_conntected_find_eulerian_path_merges_two_three_grams(self):
        self.index_to_n_gram_patch.start().return_value = self.index_to_three_gram
        eulerian_path_algorithm = EulerianPath(self.small_alphabet, n=3, min_length=3, is_merging_path=False)

        y = eulerian_path_algorithm.find_eulerian_path(self.aab_aab_weights, y_length=4)

        self.assertEqual(y, 'aabb')

    def test_merge_path_two_three_gram_cycles_find_eulerian_path_merges_two_cycles(self):
        self.index_to_n_gram_patch.start().return_value = self.index_to_three_gram
        eulerian_path_algorithm = EulerianPath(self.small_alphabet, n=3, min_length=3, is_merging_path=True)

        y = eulerian_path_algorithm.find_eulerian_path(self.aaa_bbb_weights, y_length=6)

        self.assertIn(y, self.aaaabb_bbbbaa)

    def test_smaller_length_find_eulerian_path_cuts_y_to_fit_length(self):
        y = self.eulerian_path_algorithm.find_eulerian_path(self.ab_ca_weights, y_length=2)

        self.assertIn(y, ['ab', 'ca'])

    def test_one_weight_above_threshold_find_eulerian_path_returns_one_n_gram(self):
        y = self.eulerian_path_algorithm.find_eulerian_path(self.ac_threshold_weights, thresholds=self.thresholds)

        self.assertEqual(y, 'ac')

    def test_two_weights_above_thresholds_find_eulerian_path_returns_two_n_grams(self):
        y = self.eulerian_path_algorithm.find_eulerian_path(self.aa_ac_threshold_weights, thresholds=self.thresholds)

        self.assertEqual(y, 'aac')

    def test_zero_weights_above_thresholds_find_eulerian_path_returns_maximum_weight_n_gram(self):
        y = self.eulerian_path_algorithm.find_eulerian_path(self.ac_threshold_weights, thresholds=self.high_tresholds)

        self.assertEqual(y, 'ac')

    def test_zero_weights_above_thresholds_min_length_is_three_find_eulerian_path_returns_maximum_weight_n_grams(self):
        eulerian_path_algorithm = EulerianPath(self.alphabet, n=2, min_length=3, is_merging_path=False)

        y = eulerian_path_algorithm.find_eulerian_path(self.ac_threshold_weights, thresholds=self.thresholds)

        self.assertIn(y, self.cac_aca)

    def test_one_weight_above_thresholds_min_length_is_three_find_eulerian_path_returns_maximum_weight_n_grams(self):
        eulerian_path_algorithm = EulerianPath(self.alphabet, n=2, min_length=3, is_merging_path=False)

        y = eulerian_path_algorithm.find_eulerian_path(self.ac_threshold_weights, thresholds=self.high_tresholds)

        self.assertIn(y, self.cac_aca)

    def test_rounded_weights_is_exact_length_find_eulerian_path_returns_expected_path(self):
        y = self.eulerian_path_algorithm.find_eulerian_path(self.bc_cc_non_rounded_weights, y_length=4)

        self.assertEqual(y, 'bccc')

    def test_rounded_weights_miss_one_weight_find_eulerian_path_returns_expected_path(self):
        y = self.eulerian_path_algorithm.find_eulerian_path(self.bb_bc_non_rounded_weights, y_length=3)

        self.assertEqual(y, 'bbc')

    def test_rounded_weights_miss_one_weight_find_eulerian_path_repeats_best_n_gram(self):
        y = self.eulerian_path_algorithm.find_eulerian_path(self.bb_bb_non_rounded_weights, y_length=3)

        self.assertEqual(y, 'bbb')

    def test_one_gram_eulerian_path_algorithm_raises_n_gram_length_error(self):
        with self.assertRaises(InvalidNGramLengthError):
            EulerianPath(self.alphabet, n=1)

    def test_y_length_smaller_than_n_find_eulerian_path_raises_length_error(self):
        with self.assertRaises(InvalidYLengthError):
            self.eulerian_path_algorithm.find_eulerian_path(self.ab_ba_bb_weights, y_length=1)

    def test_n_gram_weights_wrong_shape_find_eulerian_path_raises_shape_error(self):
        with self.assertRaises(InvalidShapeError):
            self.eulerian_path_algorithm.find_eulerian_path(self.a_b_c_weights, y_length=2)

    def test_no_length_no_thresholds_find_eulerian_path_raises_threshold_error(self):
        with self.assertRaises(NoThresholdsError):
            self.eulerian_path_algorithm.find_eulerian_path(self.ab_ba_bb_weights, y_length=None, thresholds=None)

    def test_thresholds_wrong_shape_find_eulerian_path_raises_shape_error(self):
        with self.assertRaises(InvalidShapeError):
            self.eulerian_path_algorithm.find_eulerian_path(self.ab_ba_bb_weights, thresholds=self.wrong_thresholds)


if __name__ == '__main__':
    unittest2.main()