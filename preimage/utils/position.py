__author__ = 'amelie'

import numpy


def compute_position_weights(position_index, max_position, sigma_position):
    position_penalties = numpy.array([(position_index - j) ** 2 for j in range(max_position)], dtype=numpy.float)
    position_penalties /= -2. * (sigma_position ** 2)
    return numpy.exp(position_penalties)


def compute_position_weights_matrix(max_position, sigma_position):
    Position_penalties = numpy.array([(i - j) for i in range(max_position) for j in range(max_position)])
    Position_penalties = Position_penalties.reshape(max_position, max_position)
    Position_penalties = numpy.square(Position_penalties) / (-2 * (sigma_position ** 2))
    return numpy.exp(Position_penalties)