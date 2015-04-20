__author__ = 'amelie'

import unittest2
import numpy.testing

from preimage.utils.position import compute_position_weights


class TestPosition(unittest2.TestCase):
    def setUp(self):
        self.position_weights_one_sigma_p_index_zero = [1., 0.6065307, 0.1353353]
        self.position_weights_one_sigma_p_index_one = [0.6065307, 1., 0.6065307]
        self.position_weights_half_sigma_p_index_zero = [1., 0.1353353, 0.0003355]

    def test_one_sigma_p_index_zero_to_one_compute_position_weights_returns_expected_weights(self):
        position_weights = compute_position_weights(position_index=0, max_position=1, sigma_position=1)

        numpy.testing.assert_array_almost_equal(position_weights, [1.])

    def test_one_sigma_p_index_zero_compute_position_weights_returns_expected_weights(self):
        position_weights = compute_position_weights(position_index=0, max_position=3, sigma_position=1)

        numpy.testing.assert_array_almost_equal(position_weights, self.position_weights_one_sigma_p_index_zero)

    def test_one_sigma_p_index_zero_compute_position_weights_returns_expected_weights(self):
        position_weights = compute_position_weights(position_index=1, max_position=3, sigma_position=1)

        numpy.testing.assert_array_almost_equal(position_weights, self.position_weights_one_sigma_p_index_one)

    def test_half_sigma_p_index_zero_compute_position_weights_returns_expected_weights(self):
        position_weights = compute_position_weights(position_index=0, max_position=3, sigma_position=0.5)

        numpy.testing.assert_array_almost_equal(position_weights, self.position_weights_half_sigma_p_index_zero)


if __name__ == '__main__':
    unittest2.main()