"""Structured output metrics"""

__author__ = 'amelie'

import numpy
from jellyfish import levenshtein_distance


def zero_one_loss(Y_true, Y_predicted):
    """Zero one loss.

    Returns the number of incorrectly predicted strings on the total number of strings.

    Parameters
    ----------
    Y_true : array, shape = [n_samples, ]
        Ground truth (correct) strings.
    Y_predicted : array, shape = [n_samples, ]
        Predicted strings, as returned by the structured output predictor.

    Returns
    -------
    loss : float
        The number of incorrectly predicted strings on the total number of strings.
    """
    Y_true = numpy.array(Y_true)
    Y_predicted = numpy.array(Y_predicted)
    __check_same_number_of_y(Y_true, Y_predicted)
    n_errors = __get_n_errors_for_each_y_predicted(Y_true, Y_predicted)
    loss = numpy.mean(n_errors)
    return loss


def __get_n_errors_for_each_y_predicted(Y_true, Y_predicted):
    n_errors = [y_true != y_predicted for (y_true, y_predicted) in zip(Y_true, Y_predicted)]
    return numpy.array(n_errors)


def hamming_loss(Y_true, Y_predicted):
    """Average hamming loss.

    Computes the average fraction of incorrectly predicted letters in the predicted strings. The ground truth and the
    predicted strings must have the same length.

    Parameters
    ----------
    Y_true : array, shape = [n_samples, ]
        Ground truth (correct) strings.
    Y_predicted : array, shape = [n_samples, ]
        Predicted strings, as returned by the structured output predictor.

    Returns
    -------
    loss : float
        The average fraction of incorrectly predicted letters.
    """
    Y_true = numpy.array(Y_true)
    Y_predicted = numpy.array(Y_predicted)
    __check_same_number_of_y(Y_true, Y_predicted)
    y_true_lengths = __get_length_of_each_y(Y_true)
    y_predicted_lengths = __get_length_of_each_y(Y_predicted)
    __check_each_tuple_y_true_y_predicted_has_same_length(y_true_lengths, y_predicted_lengths)
    n_errors = __get_n_letter_errors_for_each_y_predicted(Y_true, Y_predicted, y_true_lengths)
    loss = numpy.mean(n_errors / numpy.array(y_true_lengths, dtype=numpy.float))
    return loss


def __get_length_of_each_y(Y):
    y_lengths = numpy.array([len(y) for y in Y])
    return y_lengths


def __check_each_tuple_y_true_y_predicted_has_same_length(y_true_lengths, y_predicted_lengths):
    if not numpy.array_equal(y_true_lengths, y_predicted_lengths):
        raise ValueError('Each tuple (y_true, y_predicted) must have the same length ')


def __get_n_letter_errors_for_each_y_predicted(Y_true, Y_predicted, y_lengths):
    n_errors = [sum([y_predicted[i] != y_true[i] for i in range(y_lengths[index])])
                for index, (y_predicted, y_true) in enumerate(zip(Y_predicted, Y_true))]
    return numpy.array(n_errors)


def levenshtein_loss(Y_true, Y_predicted):
    """Average levenshtein loss.

    Computes the average fraction of levenshtein distance between the ground truth strings and the predicted strings.

    Parameters
    ----------
    Y_true : array, shape = [n_samples, ]
        Ground truth (correct) strings.
    Y_predicted : array, shape = [n_samples, ]
        Predicted strings, as returned by the structured output predictor.

    Returns
    -------
    loss : float
        The average fraction of levenshtein distance.
    """
    Y_true = numpy.array(Y_true)
    Y_predicted = numpy.array(Y_predicted)
    __check_same_number_of_y(Y_true, Y_predicted)
    max_lengths = __get_max_length_of_each_tuple_y_true_y_predicted(Y_true, Y_predicted)
    distances = __get_levenshtein_distance_for_each_y_predicted(Y_true, Y_predicted)
    loss = numpy.mean(distances / numpy.array(max_lengths, dtype=numpy.float))
    return loss


def __check_same_number_of_y(Y_true, Y_predicted):
    if Y_true.shape[0] != Y_predicted.shape[0]:
        raise ValueError('Number of Y_true must equal number of Y_predicted.'
                         'Got {:d} Y_true, {:d} Y_predicted'.format(Y_true.shape[0], Y_predicted.shape[0]))


def __get_max_length_of_each_tuple_y_true_y_predicted(Y_true, Y_predicted):
    max_lengths = [max(len(y_true), len(y_predicted)) for (y_true, y_predicted) in zip(Y_true, Y_predicted)]
    return numpy.array(max_lengths)


def __get_levenshtein_distance_for_each_y_predicted(Y_true, Y_predicted):
    distances = [levenshtein_distance(y_true, y_predicted) for (y_true, y_predicted) in zip(Y_true, Y_predicted)]
    return numpy.array(distances)