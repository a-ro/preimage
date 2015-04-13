__author__ = 'amelie'

import pickle
from os.path import dirname
from os.path import join

import numpy


class Dataset:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.y_lengths = numpy.array([len(y) for y in Y])


def load_pickle_file(file_name):
    module_path = dirname(__file__)
    data_file = open(join(module_path, file_name), 'rb')
    data = pickle.load(data_file)
    return data


def load_ocr_letters(fold_id=0):
    data = load_pickle_file('ocrletters.pickle')
    train_indexes = numpy.where(data['fold_ids'] == fold_id)
    test_indexes = numpy.where(data['fold_ids'] != fold_id)
    train_dataset = Dataset(data['X'][train_indexes], data['y'][train_indexes])
    test_dataset = Dataset(data['X'][test_indexes], data['y'][test_indexes])
    return train_dataset, test_dataset