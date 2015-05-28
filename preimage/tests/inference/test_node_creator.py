__author__ = 'amelie'

from math import sqrt

from mock import Mock
import unittest2
import numpy

from preimage.inference._bound_calculator import BoundCalculatorMock
from preimage.inference._node import MaxNode
from preimage.inference._node_creator import NodeCreator


class TestNodeCreator(unittest2.TestCase):
    def setUp(self):
        self.setup_alphabet()
        self.setup_node_creator_one_gram()
        self.setup_node_creator_two_gram()
        self.setup_nodes()

    def setup_alphabet(self):
        self.one_grams = ['a', 'b', 'c']
        self.two_grams = ['aa', 'ab', 'ba', 'bb']

    def setup_node_creator_one_gram(self):
        min_bound_calculator_mock = Mock()
        min_bound_calculator_mock.get_start_node_real_values.return_value = numpy.array([1, 2, 3], dtype=numpy.float64)
        min_bound_calculator_mock.get_start_node_bounds.return_value = numpy.array([4, 5, 6], dtype=numpy.float64)
        min_bound_calculator_mock.compute_bound.return_value = {'real_value': 1, 'bound_value': 2}
        min_bound_calculator = BoundCalculatorMock(min_bound_calculator_mock)
        max_bound_calculator_mock = Mock()
        max_bound_calculator_mock.get_start_node_real_values.return_value = numpy.array([7, 8, 9], dtype=numpy.float64)
        max_bound_calculator_mock.get_start_node_bounds.return_value = numpy.array([10, 11, 12], dtype=numpy.float64)
        max_bound_calculator_mock.compute_bound.return_value = {'real_value': 3, 'bound_value': 4}
        max_bound_calculator = BoundCalculatorMock(max_bound_calculator_mock)
        self.node_creator_one_gram = NodeCreator(min_bound_calculator, max_bound_calculator, self.one_grams)

    def setup_node_creator_two_gram(self):
        min_bound_calculator_mock = Mock()
        min_bound_calculator_mock.get_start_node_real_values.return_value = numpy.array([1, 3, 5, 7],
                                                                                        dtype=numpy.float64)
        min_bound_calculator_mock.get_start_node_bounds.return_value = numpy.array([9, 11, 13, 15], dtype=numpy.float64)
        min_bound_calculator_mock.compute_bound.return_value = {'real_value': 0.1, 'bound_value': 0.2}
        min_bound_calculator = BoundCalculatorMock(min_bound_calculator_mock)
        max_bound_calculator_mock = Mock()
        max_bound_calculator_mock.get_start_node_real_values.return_value = numpy.array([2, 4, 6, 8],
                                                                                        dtype=numpy.float64)
        max_bound_calculator_mock.get_start_node_bounds.return_value = numpy.array([10, 12, 14, 16],
                                                                                   dtype=numpy.float64)
        max_bound_calculator_mock.compute_bound.return_value = {'real_value': 0.3, 'bound_value': 0.4}
        max_bound_calculator = BoundCalculatorMock(max_bound_calculator_mock)
        self.node_creator_two_gram = NodeCreator(min_bound_calculator, max_bound_calculator, self.two_grams)

    def setup_nodes(self):
        self.a_b_c_nodes_length_one = [MaxNode('a', 10. / sqrt(4), 1., 7.), MaxNode('b', 11. / sqrt(5), 2., 8.),
                                       MaxNode('c', 12. / sqrt(6), 3., 9.)]
        self.aa_ab_ba_bb = [MaxNode('aa', 10. / sqrt(9), 1, 2), MaxNode('ab', 12 / sqrt(11), 3, 4),
                            MaxNode('ba', 14. / sqrt(13), 5, 6), MaxNode('bb', 16 / sqrt(15), 7, 8)]
        self.aa_parent_node = MaxNode('a', 5. / sqrt(2), 1, 2)
        self.aa_node_one_gram = MaxNode('aa', 4. / sqrt(2), 1, 3)
        self.abc_parent_node = MaxNode('bc', 2., 1, 3)
        self.abc_node_two_gram = MaxNode('abc', 0.4 / sqrt(0.2), 0.1, 0.3)

    def test_one_gram_length_one_get_start_nodes_returns_expected_nodes(self):
        start_nodes = self.node_creator_one_gram.get_start_nodes_python(1)

        for node_index, start_node in enumerate(start_nodes):
            with self.subTest(y=start_node.y):
                self.assertEqual(start_node, self.a_b_c_nodes_length_one[node_index],
                                 msg=self.get_message(start_node, self.a_b_c_nodes_length_one[node_index]))

    def test_two_gram_length_three_get_start_nodes_returns_expected_nodes(self):
        start_nodes = self.node_creator_two_gram.get_start_nodes_python(3)

        for node_index, start_node in enumerate(start_nodes):
            with self.subTest(y=start_node.y):
                self.assertEqual(start_node, self.aa_ab_ba_bb[node_index],
                                 msg=self.get_message(start_node, self.aa_ab_ba_bb[node_index]))

    def test_one_gram_create_node_returns_expected_node(self):
        length = 3
        node = self.node_creator_one_gram.create_node_python('aa', self.aa_parent_node, length)

        self.assertEqual(node, self.aa_node_one_gram,
                         msg=self.get_message(node, self.aa_node_one_gram))

    def test_two_gram_create_node_returns_expected_node(self):
        length = 4
        node = self.node_creator_two_gram.create_node_python('abc', self.abc_parent_node, length)

        self.assertEqual(node, self.abc_node_two_gram,
                         msg=self.get_message(node, self.abc_node_two_gram))

    def get_message(self, actual, expected):
        return "{} != {}".format(str(actual), str(expected))


if __name__ == '__main__':
    unittest2.main()