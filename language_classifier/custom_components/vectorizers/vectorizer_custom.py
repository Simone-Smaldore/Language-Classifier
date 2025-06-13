"""Module containing the abstract VectorizerCustom class."""

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable

from scipy.sparse import csr_matrix


class VectorizerCustom(ABC):
    """
    Abstract base class for a custom text vectorizer.

    This class provides the basic structure for creating vectorizers that
    transform text documents into sparse matrices,
    maintaining a vocabulary of terms.

    Attributes:
        vocab (dict): Dictionary mapping words to unique integer indices.
        vocab_size (int): Number of elements in the vocabulary.

    """

    def __init__(self) -> None:
        """
        Initialize the vectorizer with an empty vocabulary.

        Sets up an empty dictionary to map words to indices and
        initializes the vocabulary size to zero.
        """
        self.vocab = {}
        self.vocab_size = 0

    @abstractmethod
    def fit_transform(self, input_phrases: Iterable[str]) -> csr_matrix:
        """
        Fit the vectorizer to the input_phrases and transform the input_phrases into a sparse matrix.

        This method learns the vocabulary dictionary from the input_phrases and then
        transforms the input_phrases into a sparse matrix representation.

        Args:
            input_phrases (Iterable[str]): An iterable of text documents to fit and transform.

        Returns:
            csr_matrix: A compressed sparse row matrix representing the transformed input_phrases.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        """

    @abstractmethod
    def transform(self, input_phrases: Iterable[str]) -> csr_matrix:
        """
        Transform the input_phrases into a sparse matrix using an existing vocabulary.

        This method uses the vocabulary fitted by `fit_transform` to transform
        new text documents into the sparse matrix representation.

        Args:
            input_phrases (Iterable[str]): An iterable of text documents to transform.

        Returns:
            csr_matrix: A compressed sparse row matrix representing the transformed input_phrases.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        """

    def get_feature_names_out(self) -> list[str]:
        """
        Get the list of feature names ordered by their index in the vocabulary.

        This method sorts the internal vocabulary by the feature indices and
        returns the corresponding feature names in order.

        Returns:
            list[str]: A list of feature names ordered by their index.

        """
        sorted_vocab = sorted(self.vocab.items(), key=lambda x: x[1])
        return [word for word, idx in sorted_vocab]

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize the input text into a list of tokens (words).

        This method splits the input string into tokens based on word boundaries,
        extracting sequences of word characters.

        Args:
            text (str): The input text string to tokenize.

        Returns:
            list[str]: A list of token strings extracted from the text.

        """
        return re.findall(r"\b\w+\b", text)
