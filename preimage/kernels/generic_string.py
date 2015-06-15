__author__ = 'amelie'

import numpy

from preimage.datasets.loader import load_amino_acids_and_descriptors
from preimage.kernels._generic_string import element_wise_generic_string_kernel, generic_string_kernel_with_sigma_c
from preimage.kernels._generic_string import element_wise_generic_string_kernel_with_sigma_c
from preimage.datasets.amino_acid_file import AminoAcidFile
from preimage.utils.position import compute_position_weights_matrix
from preimage.utils.alphabet import transform_strings_to_integer_lists


def element_wise_kernel(X, sigma_position, n, alphabet):
    """Compute the similarity of each string in X with itself in the Generic String kernel.

    Takes only in account the position penalties and the n-gram of length n. No n-gram penalties (no sigma_c).

    Parameters
    ----------
    X : array, shape = [n_samples]
        Strings, where n_samples is the number of examples in X.
    sigma_position : float
        Controls the penalty incurred when two n-grams are not sharing the same position.
    n : int
        N-gram length.
    alphabet : list
        List of letters.

    Returns
    -------
    kernel : array, shape = [n_samples]
        Similarity of each string with itself in the GS kernel, where n_samples is the number of examples in X.
        """
    X = numpy.array(X)
    x_lengths = numpy.array([len(x) for x in X], dtype=numpy.int64)
    max_length = numpy.max(x_lengths) - n + 1
    position_matrix = compute_position_weights_matrix(max_length, sigma_position)
    X_int = transform_strings_to_integer_lists(X, alphabet)
    kernel = element_wise_generic_string_kernel(X_int, x_lengths, position_matrix, n)
    return kernel


class GenericStringKernel:
    """Generic String Kernel.

    Computes the similarity between two strings by comparing each of their l-gram of length 1 to n. Each l-gram
    comparison yields a score that depends on the similarity of their respective amino acids (letters) and a shifting
    contribution term that decays exponentially rapidly with the distance between the starting positions of the two
    substrings. The sigma_position parameter controls the shifting contribution term. The sigma_amino_acid parameter
    controls the amount of penalty incurred when the encoding vectors differ as measured by the squared Euclidean
    distance between these two vectors. The GS kernel outputs the sum of all the l-gram-comparison scores.

    Attributes
    ----------
    amino_acid_file_name : string
        Name of the file containing the amino acid substitution matrix.
    sigma_position : float
        Controls the penalty incurred when two n-grams are not sharing the same position.
    sigma_amino_acid : float
        Controls the penalty incurred when the encoding vectors of two amino acids differ.
    n : int
        N-gram length.
    is_normalized : bool
        True if the kernel should be normalized, False otherwise.

    Notes
    -----
    See http://graal.ift.ulaval.ca/bioinformatics/gs-kernel/ for the original code developed by Sebastien Giguere [1]_.

    References
    ----------
    .. [1] Sebastien Giguere, Mario Marchand, Francois Laviolette, Alexandre Drouin, and Jacques Corbeil. "Learning a
       peptide-protein binding affinity predictor with kernel ridge regression." BMC bioinformatics 14, no. 1 (2013):
       82.
    """
    def __init__(self, amino_acid_file_name=AminoAcidFile.blosum62_natural, sigma_position=1.0, sigma_amino_acid=1.0,
                 n=2, is_normalized=True):
        self.amino_acid_file_name = amino_acid_file_name
        self.sigma_position = sigma_position
        self.sigma_amino_acid = sigma_amino_acid
        self.n = n
        self.is_normalized = is_normalized
        self.alphabet, self.descriptors = self._load_amino_acids_and_normalized_descriptors()

    def __call__(self, X1, X2):
        """Compute the similarity of all the strings of X1 with all the strings of X2 in the Generic String Kernel.

        Parameters
        ----------
        X1 : array, shape=[n_samples, ]
            Strings, where n_samples is the number of samples in X1.
        X2 : array, shape=[n_samples, ]
            Strings, where n_samples is the number of samples in X2.

        Returns
        -------
        gram_matrix : array, shape = [n_samples_x1, n_samples_x2]
            Similarity of each string of X1 with each string of X2, n_samples_x1 is the number of samples in X1 and
            n_samples_x2 is the number of samples in X2.
        """
        X1 = numpy.array(X1)
        X2 = numpy.array(X2)
        amino_acid_similarity_matrix = self.get_alphabet_similarity_matrix()
        is_symmetric = bool(X1.shape == X2.shape and numpy.all(X1 == X2))
        max_length, x1_lengths, x2_lengths = self._get_lengths(X1, X2)
        position_matrix = self.get_position_matrix(max_length)
        X1_int = transform_strings_to_integer_lists(X1, self.alphabet)
        X2_int = transform_strings_to_integer_lists(X2, self.alphabet)
        gram_matrix = generic_string_kernel_with_sigma_c(X1_int, x1_lengths, X2_int, x2_lengths, position_matrix,
                                                         amino_acid_similarity_matrix, self.n, is_symmetric)
        gram_matrix = self._normalize(gram_matrix, X1_int, x1_lengths, X2_int, x2_lengths, position_matrix,
                                      amino_acid_similarity_matrix, is_symmetric)
        return gram_matrix

    def get_position_matrix(self, max_length):
        """Compute the position similarity weights

        Parameters
        ----------
        max_length : int
            Maximum position.

        Returns
        -------
        position_matrix : array, shape = [max_length, max_length]
            Similarity of each position with all the other positions.
        """
        position_matrix = compute_position_weights_matrix(max_length, self.sigma_position)
        return position_matrix

    def get_alphabet_similarity_matrix(self):
        """Compute the alphabet similarity weights

        Returns
        -------
        similarity_matrix : array, shape = [len(alphabet), len(alphabet)]
            Similarity of each amino acid (letter) with all the other amino acids.
        """
        distance_matrix = numpy.zeros((len(self.alphabet), len(self.alphabet)))
        numpy.fill_diagonal(distance_matrix, 0)
        for index_one, descriptor_one in enumerate(self.descriptors):
            for index_two, descriptor_two in enumerate(self.descriptors):
                distance = descriptor_one - descriptor_two
                squared_distance = numpy.dot(distance, distance)
                distance_matrix[index_one, index_two] = squared_distance
        distance_matrix /= 2. * (self.sigma_amino_acid ** 2)
        return numpy.exp(-distance_matrix)

    def _load_amino_acids_and_normalized_descriptors(self):
        amino_acids, descriptors = load_amino_acids_and_descriptors(self.amino_acid_file_name)
        normalization = numpy.array([numpy.dot(descriptor, descriptor) for descriptor in descriptors],
                                    dtype=numpy.float)
        normalization = normalization.reshape(-1, 1)
        descriptors /= numpy.sqrt(normalization)
        return amino_acids, descriptors

    def _get_lengths(self, X1, X2):
        x1_lengths = numpy.array([len(x) for x in X1], dtype=numpy.int64)
        x2_lengths = numpy.array([len(x) for x in X2], dtype=numpy.int64)
        max_length = max(numpy.max(x1_lengths), numpy.max(x2_lengths))
        return max_length, x1_lengths, x2_lengths

    def _normalize(self, gram_matrix, X1, x1_lengths, X2, x2_lengths, position_matrix, similarity_matrix, is_symmetric):
        if self.is_normalized:
            if is_symmetric:
                x1_norm = gram_matrix.diagonal()
                x2_norm = x1_norm
            else:
                x1_norm = element_wise_generic_string_kernel_with_sigma_c(X1, x1_lengths, position_matrix,
                                                                          similarity_matrix, self.n)
                x2_norm = element_wise_generic_string_kernel_with_sigma_c(X2, x2_lengths, position_matrix,
                                                                          similarity_matrix, self.n)
            gram_matrix = ((gram_matrix / numpy.sqrt(x2_norm)).T / numpy.sqrt(x1_norm)).T
        return gram_matrix

    def element_wise_kernel(self, X):
        """Compute the similarity of each string of X with itself in the Generic String kernel.

        Parameters
        ----------
        X : array, shape = [n_samples]
            Strings, where n_samples is the number of examples in X.

        Returns
        -------
        kernel : array, shape = [n_samples]
            Similarity of each string with itself in the GS kernel, where n_samples is the number of examples in X.
        """
        X = numpy.array(X)
        X_int = transform_strings_to_integer_lists(X, self.alphabet)
        x_lengths = numpy.array([len(x) for x in X], dtype=numpy.int64)
        max_length = numpy.max(x_lengths)
        similarity_matrix = self.get_alphabet_similarity_matrix()
        position_matrix = self.get_position_matrix(max_length)
        kernel = element_wise_generic_string_kernel_with_sigma_c(X_int, x_lengths, position_matrix, similarity_matrix,
                                                                 self.n)
        return kernel