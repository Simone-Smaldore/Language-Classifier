"""Module containing the CountVectorizerCustom class."""

from collections import defaultdict
from collections.abc import Iterable

from scipy.sparse import csr_matrix

from language_classifier.custom_components.vectorizers.vectorizer_custom import (
    VectorizerCustom,
)


class CountVectorizerCustom(VectorizerCustom):
    """
    Count vectorizer implementation extending VectorizerCustom.

    This class converts a collection of text documents into sparse matrices
    where each entry represents the count of a term in a document. The vocabulary
    is built from the input phrases during fitting.
    """

    def __init__(self) -> None:
        """
        Initialize the count vectorizer.

        Calls the superclass initializer to set up the vocabulary and its size.
        """
        super().__init__()

    def fit_transform(self, input_phrases: Iterable[str]) -> csr_matrix:
        """
        Learn the vocabulary from the input phrases and return term frequency vectors.

        Processes the input documents to build the vocabulary and counts the occurrences
        of each term in every document, returning a sparse matrix representation.

        Args:
            input_phrases (Iterable[str]): Iterable of text documents to fit and transform.

        Returns:
            csr_matrix: Sparse matrix of term counts with shape (n_docs, vocab_size).

        """
        rows, cols, data = [], [], []
        vocab = defaultdict(lambda: len(vocab))
        for i, doc in enumerate(input_phrases):
            word_counts = defaultdict(int)
            for token in self._tokenize(doc):
                word_id = vocab[token]
                word_counts[word_id] += 1
            for word_id, count in word_counts.items():
                data.append(count)
                rows.append(i)
                cols.append(word_id)
        self.vocab = dict(vocab)
        self.vocab_size = len(self.vocab)
        return csr_matrix(
            (data, (rows, cols)),
            shape=(len(input_phrases), self.vocab_size),
        )

    def transform(self, input_phrases: Iterable[str]) -> csr_matrix:
        """
        Transform new documents to sparse term frequency vectors using the learned vocabulary.

        Converts new documents into sparse matrices where entries correspond to term counts,
        based on the vocabulary built during fitting.

        Args:
            input_phrases (Iterable[str]): Iterable of text documents to transform.

        Returns:
            csr_matrix: Sparse matrix of term counts with shape (n_docs, vocab_size).

        """
        rows, cols, data = [], [], []

        for i, doc in enumerate(input_phrases):
            word_counts = defaultdict(int)
            for token in self._tokenize(doc):
                if token in self.vocab:
                    word_id = self.vocab[token]
                    word_counts[word_id] += 1
            for word_id, count in word_counts.items():
                data.append(count)
                rows.append(i)
                cols.append(word_id)

        return csr_matrix(
            (data, (rows, cols)),
            shape=(len(input_phrases), self.vocab_size),
        )
