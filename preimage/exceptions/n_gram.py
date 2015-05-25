__author__ = 'amelie'


class InvalidNGramError(ValueError):
    def __init__(self, n, n_gram):
        self.n = n
        self.n_gram = n_gram

    def __str__(self):
        error_message = "{} is not a possible {:d}_gram for this alphabet".format(self.n_gram, self.n)
        return error_message


class InvalidNGramLengthError(ValueError):
    def __init__(self, n, min_n=0):
        self.n = n
        self.min_n = min_n

    def __str__(self):
        error_message = 'n must be greater than {:d}. Got: n={:d}'.format(self.min_n, self.n)
        return error_message


class InvalidYLengthError(ValueError):
    def __init__(self, n, y_length):
        self.n = n
        self.y_length = y_length

    def __str__(self):
        error_message = 'y_length must be >= n. Got: y_length={:d}, n={:d}'.format(self.y_length, self.n)
        return error_message


class InvalidMinLengthError(ValueError):
    def __init__(self, min_length, max_length):
        self.min_length = min_length
        self.max_length = max_length

    def __str__(self):
        error_message = 'min_length must be <= max_length. ' \
                        'Got: min_length={:d}, max_length={:d}'.format(self.min_length, self.max_length)
        return error_message


class NoThresholdsError(ValueError):
    def __str__(self):
        error_message = 'thresholds must be provided when y_length is None'
        return error_message