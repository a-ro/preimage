__author__ = 'amelie'

import pickle
import gzip
from os.path import dirname, join

import numpy


class StructuredOutputDataset:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.y_lengths = numpy.array([len(y) for y in Y])


class StandardDataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y


def load_ocr_letters(fold_id=0):
    data = __load_gz_pickle_file('ocrletters.pickle.gz')
    train_indexes = numpy.where(data['fold_ids'] == fold_id)
    test_indexes = numpy.where(data['fold_ids'] != fold_id)
    Y_train = numpy.array(data['y'], dtype=numpy.str)[train_indexes]
    Y_test = numpy.array(data['y'], dtype=numpy.str)[test_indexes]
    train_dataset = StructuredOutputDataset(data['X'][train_indexes], Y_train)
    test_dataset = StructuredOutputDataset(data['X'][test_indexes], Y_test)
    return train_dataset, test_dataset


def __load_gz_pickle_file(file_name):
    module_path = dirname(__file__)
    gzip_reader = gzip.open(join(module_path, file_name), 'rb')
    data = pickle.loads(gzip_reader.read())
    return data


def load_camps_dataset():
    return __load_peptide_dataset('camps.pickle')


def load_bpps_dataset():
    return __load_peptide_dataset('bpps.pickle')


def __load_peptide_dataset(file_name):
    data = __load_pickle_file(file_name)
    X = numpy.array(data['X'], dtype=numpy.str)
    train_dataset = StandardDataset(X, data['y'])
    return train_dataset


def __load_pickle_file(file_name):
    module_path = dirname(__file__)
    data_file = open(join(module_path, file_name), 'rb')
    data = pickle.load(data_file)
    return data