__author__ = 'amelie'

from preimage.datasets.loader import load_ocr_letters
from preimage.learners.structured_krr import StructuredKernelRidgeRegression
from preimage.kernels.polynomial import PolynomialKernel
from preimage.models.eulerian_path_model import EulerianPathModel
from preimage.utils.alphabet import Alphabet
from preimage.metrics.structured_output import zero_one_loss, hamming_loss, levenshtein_loss


if __name__ == '__main__':
    # You should find the best parameters with cross-validation
    poly_kernel = PolynomialKernel(degree=2)
    n_predictions = 50
    is_using_length = True
    n = 3
    if is_using_length:
        alpha = 1e-5
    else:
        alpha = 10  # use larger alpha value when no length otherwise the predicted string lengths are too big

    print('Eulerian path model on OCR Letter Dataset')
    train_dataset, test_dataset = load_ocr_letters(fold_id=0)
    test_dataset.X = test_dataset.X[0:n_predictions]
    test_dataset.Y = test_dataset.Y[0:n_predictions]

    inference_model = EulerianPathModel(Alphabet.latin, n, is_using_length)
    learner = StructuredKernelRidgeRegression(alpha, poly_kernel, inference_model)

    print('training ...')
    learner.fit(train_dataset.X, train_dataset.Y, train_dataset.y_lengths)

    print('predict ...')
    Y_predictions = learner.predict(test_dataset.X, test_dataset.y_lengths)

    print('\n')
    print('Y predictions: ', Y_predictions)
    print('Y real: ', test_dataset.Y)

    print('\n')
    print('Results:')
    print('zero_one_loss', zero_one_loss(test_dataset.Y, Y_predictions))
    print('levenshtein_loss', levenshtein_loss(test_dataset.Y, Y_predictions))
    if is_using_length:
        print('hamming_loss', hamming_loss(test_dataset.Y, Y_predictions))