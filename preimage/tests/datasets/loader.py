__author__ = 'amelie'

import unittest2

from preimage.datasets import loader


# This is just to ensure the pickle files were made correctly
# (not real unit testing since we use the real .pickle files)
class TestLoader(unittest2.TestCase):
    def setUp(self):
        self.ocr_n_examples_in_fold = [626, 704, 684, 698, 693, 651, 739, 717, 690, 675]
        self.ocr_n_examples_not_in_fold = [6251, 6173, 6193, 6179, 6184, 6226, 6138, 6160, 6187, 6202]
        self.camps_n_examples = 101
        self.bpps_n_examples = 31

    def test_load_ocr_letters_has_correct_number_of_x_in_fold(self):
        for fold_id in range(10):
            with self.subTest(fold_id=fold_id):
                train_dataset, test_dataset = loader.load_ocr_letters(fold_id)
                n_examples = train_dataset.X.shape[0]

                self.assertEqual(n_examples, self.ocr_n_examples_in_fold[fold_id])

    def test_load_ocr_letters_has_correct_number_of_y_in_fold(self):
        for fold_id in range(10):
            with self.subTest(fold_id=fold_id):
                train_dataset, test_dataset = loader.load_ocr_letters(fold_id)
                n_examples = train_dataset.Y.shape[0]

                self.assertEqual(n_examples, self.ocr_n_examples_in_fold[fold_id])

    def test_load_ocr_letters_has_correct_number_of_x_not_in_fold(self):
        for fold_id in range(10):
            with self.subTest(fold_id=fold_id):
                train_dataset, test_dataset = loader.load_ocr_letters(fold_id)
                n_examples = test_dataset.X.shape[0]

                self.assertEqual(n_examples, self.ocr_n_examples_not_in_fold[fold_id])

    def test_load_ocr_letters_has_correct_number_of_y_not_in_fold(self):
        for fold_id in range(10):
            with self.subTest(fold_id=fold_id):
                train_dataset, test_dataset = loader.load_ocr_letters(fold_id)
                n_examples = test_dataset.Y.shape[0]

                self.assertEqual(n_examples, self.ocr_n_examples_not_in_fold[fold_id])

    def test_load_camps_dataset_has_correct_number_of_x(self):
        dataset = loader.load_camps_dataset()
        n_examples = dataset.X.shape[0]

        self.assertEqual(n_examples, self.camps_n_examples)

    def test_load_camps_dataset_has_correct_number_of_y(self):
        dataset = loader.load_camps_dataset()
        n_examples = dataset.y.shape[0]

        self.assertEqual(n_examples, self.camps_n_examples)

    def test_load_bpps_dataset_has_correct_number_of_x(self):
        dataset = loader.load_bpps_dataset()
        n_examples = dataset.X.shape[0]

        self.assertEqual(n_examples, self.bpps_n_examples)

    def test_load_bpps_dataset_has_correct_number_of_y(self):
        dataset = loader.load_bpps_dataset()
        n_examples = dataset.y.shape[0]

        self.assertEqual(n_examples, self.bpps_n_examples)


if __name__ == '__main__':
    unittest2.main()