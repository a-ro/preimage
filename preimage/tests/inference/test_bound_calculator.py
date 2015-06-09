__author__ = 'amelie'

import unittest2
import numpy

from preimage.inference.bound_calculator import MaxBoundCalculator, OCRMinBoundCalculator


class TestMaxBoundCalculator(unittest2.TestCase):
    def setUp(self):
        self.setup_alphabet()
        self.setup_graph()
        self.setup_weights()
        self.setup_bound_calculator()
        self.setup_bounds()

    def setup_alphabet(self):
        self.one_grams = ['a', 'b', 'c']
        self.one_gram_to_index = {'a': 0, 'b': 1, 'c': 2}
        self.two_gram_to_index = {'aa': 0, 'ab': 1, 'ac': 2, 'ba': 3, 'bb': 4, 'bc': 5, 'ca': 6, 'cb': 7, 'cc': 8}

    def setup_graph(self):
        self.graph_one_gram = numpy.array([[0, 2, 1], [3, 2, 2.], [3, 3, 4]], dtype=numpy.float64)
        self.graph_one_gram_1d_weight = numpy.array([[0, 2, 1], [2, 4, 3.]], dtype=numpy.float64)
        self.graph_two_gram = numpy.array([numpy.arange(0, 9), [6, 7, 9, 10, 11, 12, 14, 15, 16]], dtype=numpy.float64)

    def setup_weights(self):
        self.weights_one_gram_1d = numpy.array([[0, 2, 1]], dtype=numpy.float64)
        self.weights_one_gram = numpy.array([[0, 2, 1], [1, 0, 0], [0, 0, 1]], dtype=numpy.float64)
        self.weights_two_gram = numpy.array(numpy.arange(0, 9).reshape(1, -1), dtype=numpy.float64)

    def setup_bound_calculator(self):
        self.bound_calculator_one_gram = MaxBoundCalculator(1, self.graph_one_gram, self.weights_one_gram,
                                                            self.one_gram_to_index)
        self.bound_calculator_1d_weight = MaxBoundCalculator(1, self.graph_one_gram_1d_weight, self.weights_one_gram_1d,
                                                             self.one_gram_to_index)
        self.bound_calculator_two_gram = MaxBoundCalculator(2, self.graph_two_gram, self.weights_two_gram,
                                                            self.two_gram_to_index)

    def setup_bounds(self):
        self.ab_length_three_bound = {'real_value': 1, 'bound_value': 3}
        self.bab_length_three_bound = {'real_value': 3, 'bound_value': 3}
        self.ac_length_two_two_gram_bound = {'real_value': 2, 'bound_value': 2}
        self.ac_length_three_two_gram_bound = {'real_value': 2, 'bound_value': 9}
        self.bac_length_three_two_gram_bound = {'real_value': 5, 'bound_value': 5}

    def test_one_gram_length_one_start_node_real_values_is_first_graph_weights_partition(self):
        length = 1

        real_values = self.bound_calculator_one_gram.get_start_node_real_values_python(length)

        numpy.testing.assert_array_almost_equal(real_values, self.weights_one_gram[0, :])

    def test_one_gram_length_one_start_nodes_bound_values_is_first_graph_partition(self):
        length = 1

        bound_values = self.bound_calculator_one_gram.get_start_node_bounds_python(length)

        numpy.testing.assert_array_almost_equal(bound_values, self.graph_one_gram[0, :])

    def test_one_gram_length_two_start_node_real_values_is_second_graph_weights_partition(self):
        length = 2

        real_values = self.bound_calculator_one_gram.get_start_node_real_values_python(length)

        numpy.testing.assert_array_almost_equal(real_values, self.weights_one_gram[1, :])

    def test_one_gram_length_two_start_node_bound_values_is_second_graph_weights_partition(self):
        length = 2

        bound_values = self.bound_calculator_one_gram.get_start_node_bounds_python(length)

        numpy.testing.assert_array_almost_equal(bound_values, self.graph_one_gram[1, :])

    def test_one_gram_length_two_1d_weights_start_node_real_values_is_first_graph_weights_partition(self):
        length = 2

        real_values = self.bound_calculator_1d_weight.get_start_node_real_values_python(length)

        numpy.testing.assert_array_almost_equal(real_values, self.weights_one_gram_1d[0, :])

    def test_two_gram_length_two_start_node_real_values_is_first_graph_weights_partition(self):
        length = 2

        real_values = self.bound_calculator_two_gram.get_start_node_real_values_python(length)

        numpy.testing.assert_array_almost_equal(real_values, self.weights_two_gram[0, :])

    def test_two_gram_length_three_start_node_bound_values_is_second_graph_partition(self):
        length = 3

        real_values = self.bound_calculator_two_gram.get_start_node_bounds_python(length)

        numpy.testing.assert_array_almost_equal(real_values, self.graph_two_gram[1, :])

    def test_one_gram_length_one_compute_bound_returns_expected_bound(self):
        length = 1

        for i, one_gram in enumerate(self.one_grams):
            with self.subTest(y=one_gram):
                bound = self.bound_calculator_one_gram.compute_bound_python(one_gram, 0, length)

                self.assertEqual(bound['real_value'], self.weights_one_gram[0, i])
                self.assertEqual(bound['bound_value'], self.graph_one_gram[0, i])

    def test_one_gram_length_three_b_compute_bound_returns_expected_bound(self):
        length = 3

        bound = self.bound_calculator_one_gram.compute_bound_python('b', 0, length)

        self.assertEqual(bound['real_value'], self.weights_one_gram[2, 1])
        self.assertEqual(bound['bound_value'], self.graph_one_gram[2, 1])


    def test_one_gram_length_three_ab_compute_bound_returns_expected_bound(self):
        length = 3

        bound = self.bound_calculator_one_gram.compute_bound_python('ab', 0, length)

        self.assertDictEqual(bound, self.ab_length_three_bound)

    def test_one_gram_length_three_aab_compute_bound_returns_expected_bound(self):
        length = 3

        bound = self.bound_calculator_one_gram.compute_bound_python('bab', self.ab_length_three_bound['real_value'],
                                                                    length)

        self.assertDictEqual(bound, self.bab_length_three_bound)

    def test_two_gram_length_two_ac_compute_bound_returns_expected_bound(self):
        length = 2

        bound = self.bound_calculator_two_gram.compute_bound_python('ac', 0, length)

        self.assertDictEqual(bound, self.ac_length_two_two_gram_bound)

    def test_two_gram_length_three_ac_compute_bound_returns_expected_bound(self):
        length = 3

        bound = self.bound_calculator_two_gram.compute_bound_python('ac', 0, length)

        self.assertDictEqual(bound, self.ac_length_three_two_gram_bound)

    def test_two_gram_length_three_bac_compute_bound_returns_expected_bound(self):
        length = 3

        bound = self.bound_calculator_two_gram.compute_bound_python('bac',
                                                                    self.ac_length_three_two_gram_bound['real_value'],
                                                                    length)

        self.assertDictEqual(bound, self.bac_length_three_two_gram_bound)


class TestOCRMinBoundCalculator(unittest2.TestCase):
    def setUp(self):
        self.setup_alphabet()
        self.setup_position_weights()
        self.setup_bound_calculator()
        self.setup_bounds()

    def setup_alphabet(self):
        self.one_grams = ['a', 'b', 'c']
        self.two_grams = ['aa', 'ab', 'ac', 'ba', 'bb', 'bc', 'ca', 'cb', 'cc']

    def setup_position_weights(self):
        self.position_weights_large_sigma = numpy.array([1, 1, 1], dtype=numpy.float64)
        self.position_weights_medium_sigma = numpy.array([1, 0.5, 0], dtype=numpy.float64)
        self.position_weights_small_sigma = numpy.array([1, 0, 0], dtype=numpy.float64)

    def setup_bound_calculator(self):
        self.bound_calculator_one_gram = OCRMinBoundCalculator(1, self.position_weights_small_sigma, self.one_grams)
        self.bound_calculator_medium_sigma = OCRMinBoundCalculator(1, self.position_weights_medium_sigma,
                                                                   self.one_grams)
        self.bound_calculator_large_sigma = OCRMinBoundCalculator(1, self.position_weights_large_sigma, self.one_grams)
        self.bound_calculator_two_gram = OCRMinBoundCalculator(2, self.position_weights_small_sigma, self.two_grams)

    def setup_bounds(self):
        self.a_length_one_bound = {'real_value': 1, 'bound_value': 1}
        self.aa_small_sigma = {'real_value': 2, 'bound_value': 2}
        self.aa_medium_sigma = {'real_value': 3, 'bound_value': 3}
        self.aa_large_sigma = {'real_value': 4, 'bound_value': 4}
        self.bb_two_gram_length_three = {'real_value': 1, 'bound_value': 2}

    def test_one_gram_length_one_start_node_real_values_is_one(self):
        length = 1

        real_values = self.bound_calculator_one_gram.get_start_node_real_values_python(length)

        numpy.testing.assert_array_almost_equal(real_values, numpy.ones(len(self.one_grams)))

    def test_one_gram_length_one_start_nodes_bound_values_is_one(self):
        length = 1

        bound_values = self.bound_calculator_one_gram.get_start_node_bounds_python(length)

        numpy.testing.assert_array_almost_equal(bound_values, numpy.ones(len(self.one_grams)))

    def test_one_gram_length_two_start_node_real_values_is_one(self):
        length = 2

        real_values = self.bound_calculator_one_gram.get_start_node_real_values_python(length)

        numpy.testing.assert_array_almost_equal(real_values, numpy.ones(len(self.one_grams)))

    def test_one_gram_length_two_start_node_bound_values_is_two(self):
        length = 2

        bound_values = self.bound_calculator_one_gram.get_start_node_bounds_python(length)

        numpy.testing.assert_array_almost_equal(bound_values, 2 * numpy.ones(len(self.one_grams)))

    def test_two_gram_length_one_start_node_bound_values_is_one(self):
        length = 2

        bound_values = self.bound_calculator_two_gram.get_start_node_bounds_python(length)

        numpy.testing.assert_array_almost_equal(bound_values, numpy.ones(len(self.two_grams)))

    def test_one_gram_length_one_compute_bound_returns_expected_bound(self):
        length = 1

        bound = self.bound_calculator_one_gram.compute_bound_python('a', 0, length)

        self.assertDictEqual(bound, self.a_length_one_bound)

    def test_one_gram_length_two_small_sigma_compute_bound_returns_expected_bound(self):
        length = 2
        parent_value = 1

        bound = self.bound_calculator_one_gram.compute_bound_python('aa', parent_value, length)

        self.assertDictEqual(bound, self.aa_small_sigma)

    def test_one_gram_length_two_medium_sigma_compute_bound_returns_expected_bound(self):
        length = 2
        parent_value = 1

        bound = self.bound_calculator_medium_sigma.compute_bound_python('aa', parent_value, length)

        self.assertDictEqual(bound, self.aa_medium_sigma)

    def test_one_gram_length_two_large_sigma_compute_bound_returns_expected_bound(self):
        length = 2
        parent_value = 1

        bound = self.bound_calculator_large_sigma.compute_bound_python('aa', parent_value, length)

        self.assertDictEqual(bound, self.aa_large_sigma)

    def test_two_gram_length_three_compute_bound_returns_expected_bound(self):
        length = 3
        parent_value = 0

        bound = self.bound_calculator_two_gram.compute_bound_python('bb', parent_value, length)

        self.assertDictEqual(bound, self.bb_two_gram_length_three)


if __name__ == '__main__':
    unittest2.main()