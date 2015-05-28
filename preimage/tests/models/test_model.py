__author__ = 'amelie'

import unittest2
import numpy
import numpy.testing

from preimage.models.model import Model
from preimage.learners.structured_krr import InferenceFitParameters
from preimage.exceptions.n_gram import NoYLengthsError


class TestModel(unittest2.TestCase):
    def setUp(self):
        self.alphabet = ['a', 'b', 'c']
        self.model_with_length = Model(self.alphabet, n=2, is_using_length=True)
        self.weights = numpy.array([[1, 2]])
        self.gram_matrix = numpy.array([[1, 0], [0, 1]])
        self.fit_parameters_no_length = InferenceFitParameters(self.weights, self.gram_matrix, Y=['a', 'ab'],
                                                               y_lengths=None)
        self.fit_parameters_with_length = InferenceFitParameters(self.weights, self.gram_matrix, Y=['a', 'ab'],
                                                                 y_lengths=[1, 2])
        self.min_length = 1
        self.max_length = 2

    def test_model_with_length_fit_has_correct_min_max_lengths(self):
        self.model_with_length.fit(self.fit_parameters_with_length)

        self.assertEqual(self.model_with_length._min_length_, self.min_length)
        self.assertEqual(self.model_with_length._max_length_, self.max_length)

    def test_model_no_y_lengths_fit_has_correct_min_max_lengths(self):
        self.model_with_length.fit(self.fit_parameters_no_length)

        self.assertEqual(self.model_with_length._min_length_, self.min_length)
        self.assertEqual(self.model_with_length._max_length_, self.max_length)

    def test_no_y_lengths_model_with_length_raises_error(self):
        self.model_with_length.fit(self.fit_parameters_with_length)

        with self.assertRaises(NoYLengthsError):
            self.model_with_length._verify_y_lengths_is_not_none_when_use_length(y_lengths=None)


if __name__ == '__main__':
    unittest2.main()