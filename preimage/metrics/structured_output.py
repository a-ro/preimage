__author__ = 'amelie'

import numpy


def zero_one_loss(Y_true, Y_predicted):
    Y_true = numpy.array(Y_true)
    Y_predicted = numpy.array(Y_predicted)
    __check_same_number_of_y(Y_true, Y_predicted)
    n_examples = Y_true.shape[0]
    errors = [Y_true[i] != Y_predicted[i] for i in range(n_examples)]
    loss = numpy.mean(errors)
    return loss


def __check_same_number_of_y(Y_true, Y_predicted):
    if Y_true.shape[0] != Y_predicted.shape[0]:
        raise ValueError('Number of Y_true must equal number of Y_predicted.'
                         'Got {:d} Y_true, {:d} Y_predicted'.format(Y_true.shape[0], Y_predicted.shape[0]))