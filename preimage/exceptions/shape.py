__author__ = 'amelie'


class InvalidShapeError(ValueError):
    def __init__(self, parameter_name, parameter_shape, valid_shapes):
        self.parameter_name = parameter_name
        self.parameter_shape = parameter_shape
        self.valid_shapes = [str(valid_shape) for valid_shape in valid_shapes]

    def __str__(self):
        valid_shapes_string = ' or '.join(self.valid_shapes)
        error_message = "{} wrong shape: Expected: {} Got: {}".format(self.parameter_name, valid_shapes_string,
                                                                      str(self.parameter_shape))
        return error_message