__author__ = 'amelie'

from sklearn.kernel_ridge import KernelRidge

from preimage.datasets.loader import load_bpps_dataset, AminoAcidFile
from preimage.kernels.generic_string import GenericStringKernel
from preimage.models.string_max_model import StringMaximizationModel


if __name__ == '__main__':
    # Best parameters found by cross-validation
    alpha = 1. / 6.4
    n = 3
    sigma_position = 0.4
    sigma_amino_acid = 0.8

    # Choose the number of predicted peptides and their length
    n_predictions = 1000
    y_length = 5

    # Max time (seconds) for the branch and bound search
    max_time = 500

    print('String maximization model on BPPs dataset')
    gs_kernel = GenericStringKernel(AminoAcidFile.blosum62_natural, sigma_position, sigma_amino_acid, n,
                                    is_normalized=True)
    alphabet = gs_kernel.alphabet
    dataset = load_bpps_dataset()

    # Use a regression algorithm to learn the weights first
    print('Learning the regression weights ...')
    learner = KernelRidge(alpha, kernel='precomputed')
    gram_matrix = gs_kernel(dataset.X, dataset.X)
    learner.fit(gram_matrix, dataset.y)
    learned_weights = learner.dual_coef_

    # We can then use the string maximization model with the learned weights
    print('Branch and bound search for the top {} peptides of length {} ...'.format(n_predictions, y_length))
    model = StringMaximizationModel(alphabet, n, gs_kernel, max_time)
    model.fit(dataset.X, learned_weights, y_length)
    peptides, bioactivities = model.predict(n_predictions)

    print('\n')
    print('Peptides | Predicted bioactivities')
    for peptide, bioactivity in zip(peptides, bioactivities):
        print(peptide, bioactivity)