__author__ = 'amelie'


class InvalidNGramError(ValueError):
    def __init__(self, n, n_gram):
        self.n = n
        self.n_gram = n_gram

    def __str__(self):
        error_message = "{} is not a possible {:d}_gram for this alphabet".format(self.n_gram, self.n)
        return error_message


class InvalidNGramLengthError(ValueError):
    def __init__(self, n):
        self.n = n

    def __str__(self):
        error_message = 'n must be greater than zero. Got: n={:d}'.format(self.n)
        return error_message