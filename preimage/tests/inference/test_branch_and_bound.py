__author__ = 'amelie'

from mock import MagicMock
import unittest2

from preimage.inference.branch_and_bound import branch_and_bound, branch_and_bound_no_length
from preimage.inference.node_creator import NodeCreatorMock
from preimage.inference.node import MaxNode


def bound_side_effect_cb_best(y, final_length):
    nodes = {'ab': MaxNode('ab', 1., 0, 0.), 'ac': MaxNode('ac', 0., 0, 0.), 'bb': MaxNode('bb', 1., 0, 0.),
             'bc': MaxNode('bc', 0, 0, 0.), 'cb': MaxNode('cb', 1.5, 0, 0.), 'cc': MaxNode('cc', 0., 0, 0.)}
    return nodes[y]


def bound_side_effect_cc_best(y, final_length):
    nodes = {'ab': MaxNode('ab', 1., 0, 0.), 'ac': MaxNode('ac', 0., 0, 0.), 'bb': MaxNode('bb', 1., 0, 0.),
             'bc': MaxNode('bc', 0, 0, 0.), 'cb': MaxNode('cb', 1.5, 0, 0.), 'cc': MaxNode('cc', 1.6, 0, 0.)}
    return nodes[y]


def bound_side_effect_acb_best(y, final_length):
    nodes = {'ab': MaxNode('ab', 1., 0, 0.), 'bb': MaxNode('bb', 1., 0, 0.), 'cb': MaxNode('cb', 1.5, 0, 0.),
             'acb': MaxNode('acb', 1.4, 0, 0.), 'bcb': MaxNode('bcb', 1.3, 0, 0.), 'ccb': MaxNode('ccb', 1.3, 0, 0.)}
    return nodes[y]


def bound_side_effect_bbb_best(y, final_length):
    nodes = {'ab': MaxNode('ab', 1., 0, 0.), 'bb': MaxNode('bb', 1.4, 0, 0.), 'cb': MaxNode('cb', 1.5, 0, 0.),
             'acb': MaxNode('acb', 1, 0, 0.), 'bcb': MaxNode('bcb', 1.1, 0, 0.), 'ccb': MaxNode('ccb', 1.1, 0, 0.),
             'abb': MaxNode('abb', 1, 0, 0.), 'bbb': MaxNode('bbb', 1.3, 0, 0.), 'cbb': MaxNode('cbb', 1.1, 0, 0.)}
    return nodes[y]


def bound_side_effect_abb_best_with_two_heap_pop(y, final_length):
    nodes = {'ab': MaxNode('ab', 1., 0, 0.), 'bb': MaxNode('bb', 1.4, 0, 0.), 'cb': MaxNode('cb', 1.5, 0, 0.),
             'acb': MaxNode('acb', 1, 0, 0.), 'bcb': MaxNode('bcb', 1.1, 0, 0.), 'ccb': MaxNode('ccb', 1.1, 0, 0.),
             'abb': MaxNode('abb', 1.3, 0, 0.), 'bbb': MaxNode('bbb', 1, 0, 0.), 'cbb': MaxNode('cbb', 1.1, 0, 0.),
             'ac': MaxNode('ac', 1., 0, 0.), 'bc': MaxNode('bc', 1, 0, 0.), 'cc': MaxNode('cb', 0, 0, 0.)}
    return nodes[y]


def bound_side_effect_bc_best_multiple_lengths(y, final_length):
    nodes = {2: {'ab': MaxNode('ab', 1., 0, 0.), 'bb': MaxNode('bb', 1.4, 0, 0.), 'cb': MaxNode('cb', 1.2, 0, 0.),
                 'ac': MaxNode('ac', 1., 0, 0.), 'bc': MaxNode('bc', 1.5, 0, 0.), 'cc': MaxNode('cc', 1.2, 0, 0.)},
             3: {'aa': MaxNode('aa', 1.6, 0, 0.), 'ba': MaxNode('ba', 1, 0, 0.), 'ca': MaxNode('ca', 1, 0, 0.),
                 'aaa': MaxNode('aaa', 1.4, 0, 0.), 'baa': MaxNode('baa', 1, 0, 0.), 'caa': MaxNode('caa', 1, 0, 0.)}}
    return nodes[final_length][y]


def start_node_side_effect_b_length_one_best(final_length):
    nodes = {1: [MaxNode('a', 0., 0., 0.), MaxNode('b', 2., 0, 0), MaxNode('c', 1., 0, 0)],
             2: [MaxNode('a', 0., 0, 0.), MaxNode('b', 1., 0, 0), MaxNode('c', 0., 0, 0)]}
    return nodes[final_length]


def start_node_side_effect_b_length_two_best(final_length):
    nodes = {1: [MaxNode('a', 0., 0, 0), MaxNode('b', 2., 0, 0.), MaxNode('c', 1., 0, 0)],
             2: [MaxNode('a', 0., 0, 0), MaxNode('b', 3., 0, 0), MaxNode('c', 0., 0, 0)]}
    return nodes[final_length]


def start_node_side_effect_b_length_two_solution_best(final_length):
    nodes = {1: [MaxNode('a', 0., 0, 0), MaxNode('b', 1., 0, 0), MaxNode('c', 1., 0, 0)],
             2: [MaxNode('a', 0., 0, 0.), MaxNode('b', 3., 0, 0.), MaxNode('c', 0., 0, 0)]}
    return nodes[final_length]


def start_node_side_effect_start_at_b_length_two(final_length):
    nodes = {1: [MaxNode('a', 0., 0, 0), MaxNode('b', 1., 0, 0), MaxNode('c', 0.5, 0, 0)],
             2: [MaxNode('a', 0., 0, 0.), MaxNode('b', 3, 0, 0), MaxNode('c', 2., 0, 0)],
             3: [MaxNode('a', 1.6, 0, 0), MaxNode('b', 0., 0, 0), MaxNode('c', 0., 0, 0)]}
    return nodes[final_length]


class TestBranchAndBound(unittest2.TestCase):
    def setUp(self):
        self.setup_alphabet()
        self.setup_nodes()
        self.time = 30

    def setup_alphabet(self):
        self.alphabet = ['a', 'b', 'c']

    def setup_bound_calculator(self, start_nodes, side_effect=bound_side_effect_cb_best):
        bound_calculator_mock = MagicMock()
        bound_calculator_mock.get_start_nodes.return_value = start_nodes
        bound_calculator_mock.get_start_nodes.return_value = start_nodes
        bound_calculator_mock.create_node.side_effect = side_effect
        self.bound_calculator = NodeCreatorMock(bound_calculator_mock)

    def setup_nodes(self):
        self.b_node_bound = 2
        self.cb_node_bound = 1.5
        self.cc_node_bound = 1.6
        self.acb_node_bound = 1.4
        self.bbb_node_bound = 1.3
        self.abb_node_bound = 1.3
        self.a_b_c_nodes = [MaxNode('a', 0., 1., 0.), MaxNode('b', 2., 1., 2.), MaxNode('c', 1., 1., 1.)]
        self.a_b_c_nodes_large_c = [MaxNode('a', 0., 0, 0.), MaxNode('b', 2., 0, 0), MaxNode('c', 1.6, 0, 0)]

    def test_one_gram_length_one_branch_and_bound_returns_max_start_node(self):
        y_length = 1
        self.setup_bound_calculator(self.a_b_c_nodes)

        y, y_bound = branch_and_bound(self.bound_calculator, y_length, self.alphabet, self.time)

        self.assertEqual(y, 'b')
        self.assertEqual(y_bound, self.b_node_bound)

    def test_one_gram_length_two_branch_and_bound_returns_best_node(self):
        y_length = 2
        self.setup_bound_calculator(self.a_b_c_nodes)

        y, y_bound = branch_and_bound(self.bound_calculator, y_length, self.alphabet, self.time)

        self.assertEqual(y, 'cb')
        self.assertEqual(y_bound, self.cb_node_bound)

    def test_one_gram_length_two_node_in_heap_better_than_first_solution_branch_and_bound_returns_best_node(self):
        y_length = 2
        self.setup_bound_calculator(self.a_b_c_nodes_large_c)

        y, y_bound = branch_and_bound(self.bound_calculator, y_length, self.alphabet, self.time)

        self.assertEqual(y, 'cb')
        self.assertEqual(y_bound, self.cb_node_bound)

    def test_one_gram_length_two_other_solution_better_than_first_solution_branch_and_bound_returns_best_node(self):
        y_length = 2
        self.setup_bound_calculator(self.a_b_c_nodes_large_c, bound_side_effect_cc_best)

        y, y_bound = branch_and_bound(self.bound_calculator, y_length, self.alphabet, self.time)

        self.assertEqual(y, 'cc')
        self.assertEqual(y_bound, self.cc_node_bound)

    def test_one_gram_length_three_branch_and_bound_returns_best_node(self):
        y_length = 3
        self.setup_bound_calculator(self.a_b_c_nodes, bound_side_effect_acb_best)

        y, y_bound = branch_and_bound(self.bound_calculator, y_length, self.alphabet, self.time)

        self.assertEqual(y, 'acb')
        self.assertEqual(y_bound, self.acb_node_bound)

    def test_one_gram_length_three_other_solution_better_than_first_three_branch_and_bound_returns_best_node(self):
        y_length = 3
        self.setup_bound_calculator(self.a_b_c_nodes, bound_side_effect_bbb_best)

        y, y_bound = branch_and_bound(self.bound_calculator, y_length, self.alphabet, self.time)

        self.assertEqual(y, 'bbb')
        self.assertEqual(y_bound, self.bbb_node_bound)

    def test_one_gram_length_three_other_nodes_better_than_solution_branch_and_bound_returns_best_node(self):
        y_length = 3
        self.setup_bound_calculator(self.a_b_c_nodes_large_c, bound_side_effect_abb_best_with_two_heap_pop)

        y, y_bound = branch_and_bound(self.bound_calculator, y_length, self.alphabet, self.time)

        self.assertEqual(y, 'abb')
        self.assertEqual(y_bound, self.abb_node_bound)


class TestBranchAndBoundNoLength(unittest2.TestCase):
    def setUp(self):
        self.setup_alphabet()
        self.setup_nodes()
        self.time = 30

    def setup_alphabet(self):
        self.alphabet = ['a', 'b', 'c']

    def setup_bound_calculator(self, start_nodes, bound_side_effect=bound_side_effect_cb_best):
        bound_calculator_mock = MagicMock()
        bound_calculator_mock.get_start_nodes.return_value = start_nodes
        bound_calculator_mock.create_node.side_effect = bound_side_effect
        self.bound_calculator = NodeCreatorMock(bound_calculator_mock)

    def setup_bound_calculator_node_side_effect(self, start_nodes, bound_side_effect=bound_side_effect_cb_best):
        self.bound_calculator_mock = MagicMock()
        self.bound_calculator_mock.get_start_nodes.side_effect = start_nodes
        self.bound_calculator_mock.create_node.side_effect = bound_side_effect
        self.bound_calculator = NodeCreatorMock(self.bound_calculator_mock)

    def setup_nodes(self):
        self.b_node_bound = 2
        self.cb_node_bound = 1.5
        self.bc_node_bound = 1.5
        self.cc_node_bound = 1.6
        self.acb_node_bound = 1.4
        self.bbb_node_bound = 1.3
        self.abb_node_bound = 1.3
        self.a_b_c_nodes = [MaxNode('a', 0., 1., 0.), MaxNode('b', 2., 1., 2.), MaxNode('c', 1., 1., 1.)]
        self.a_b_c_nodes_large_c = [MaxNode('a', 0., 0, 0.), MaxNode('b', 2., 0, 0), MaxNode('c', 1.6, 0, 0)]

    def test_one_gram_length_one_to_two_branch_and_bound_returns_max_start_node(self):
        min_length = 1
        max_length = 2
        self.setup_bound_calculator_node_side_effect(start_node_side_effect_b_length_one_best)

        y, y_bound = branch_and_bound_no_length(self.bound_calculator, min_length, max_length, self.alphabet, self.time)

        self.assertEqual(y, 'b')
        self.assertEqual(y_bound, self.b_node_bound)

    def test_length_one_to_two_length_two_better_node_branch_and_bound_returns_first_solution(self):
        min_length = 1
        max_length = 2
        self.setup_bound_calculator_node_side_effect(start_node_side_effect_b_length_two_best,
                                                     bound_side_effect_cb_best)

        y, y_bound = branch_and_bound_no_length(self.bound_calculator, min_length, max_length, self.alphabet, self.time)

        self.assertEqual(y, 'b')
        self.assertEqual(y_bound, self.b_node_bound)

    def test_length_one_to_two_length_two_better_solution_branch_and_bound_returns_length_two_solution(self):
        min_length = 1
        max_length = 2
        self.setup_bound_calculator_node_side_effect(start_node_side_effect_b_length_two_solution_best,
                                                     bound_side_effect_cb_best)

        y, y_bound = branch_and_bound_no_length(self.bound_calculator, min_length, max_length, self.alphabet, self.time)

        self.assertEqual(y, 'cb')
        self.assertEqual(y_bound, self.cb_node_bound)

    def test_length_one_to_three_length_two_better_solution_branch_and_bound_returns_length_two_solution(self):
        min_length = 1
        max_length = 3
        self.setup_bound_calculator_node_side_effect(start_node_side_effect_start_at_b_length_two,
                                                     bound_side_effect_bc_best_multiple_lengths)

        y, y_bound = branch_and_bound_no_length(self.bound_calculator, min_length, max_length, self.alphabet, self.time)

        self.assertEqual(y, 'bc')
        self.assertEqual(y_bound, self.cb_node_bound)

    def test_one_gram_length_one_branch_and_bound_returns_max_start_node(self):
        y_length = 1
        self.setup_bound_calculator(self.a_b_c_nodes)

        y, y_bound = branch_and_bound_no_length(self.bound_calculator, y_length, y_length, self.alphabet, self.time)

        self.assertEqual(y, 'b')
        self.assertEqual(y_bound, self.b_node_bound)

    def test_one_gram_length_two_branch_and_bound_returns_best_node(self):
        y_length = 2
        self.setup_bound_calculator(self.a_b_c_nodes)

        y, y_bound = branch_and_bound_no_length(self.bound_calculator, y_length, y_length, self.alphabet, self.time)

        self.assertEqual(y, 'cb')
        self.assertEqual(y_bound, self.cb_node_bound)

    def test_one_gram_length_two_node_in_heap_better_than_first_solution_branch_and_bound_returns_best_node(self):
        y_length = 2
        self.setup_bound_calculator(self.a_b_c_nodes_large_c)

        y, y_bound = branch_and_bound_no_length(self.bound_calculator, y_length, y_length, self.alphabet, self.time)

        self.assertEqual(y, 'cb')
        self.assertEqual(y_bound, self.cb_node_bound)

    def test_one_gram_length_two_other_solution_better_than_first_solution_branch_and_bound_returns_best_node(self):
        y_length = 2
        self.setup_bound_calculator(self.a_b_c_nodes_large_c, bound_side_effect_cc_best)

        y, y_bound = branch_and_bound_no_length(self.bound_calculator, y_length, y_length, self.alphabet, self.time)

        self.assertEqual(y, 'cc')
        self.assertEqual(y_bound, self.cc_node_bound)

    def test_one_gram_length_three_branch_and_bound_returns_best_node(self):
        y_length = 3
        self.setup_bound_calculator(self.a_b_c_nodes, bound_side_effect_acb_best)

        y, y_bound = branch_and_bound_no_length(self.bound_calculator, y_length, y_length, self.alphabet, self.time)

        self.assertEqual(y, 'acb')
        self.assertEqual(y_bound, self.acb_node_bound)

    def test_one_gram_length_three_other_solution_better_than_first_three_branch_and_bound_returns_best_node(self):
        y_length = 3
        self.setup_bound_calculator(self.a_b_c_nodes, bound_side_effect_bbb_best)

        y, y_bound = branch_and_bound_no_length(self.bound_calculator, y_length, y_length, self.alphabet, self.time)

        self.assertEqual(y, 'bbb')
        self.assertEqual(y_bound, self.bbb_node_bound)

    def test_one_gram_length_three_other_nodes_better_than_solution_branch_and_bound_returns_best_node(self):
        y_length = 3
        self.setup_bound_calculator(self.a_b_c_nodes_large_c, bound_side_effect_abb_best_with_two_heap_pop)

        y, y_bound = branch_and_bound_no_length(self.bound_calculator, y_length, y_length, self.alphabet, self.time)

        self.assertEqual(y, 'abb')
        self.assertEqual(y_bound, self.abb_node_bound)


if __name__ == '__main__':
    unittest2.main()