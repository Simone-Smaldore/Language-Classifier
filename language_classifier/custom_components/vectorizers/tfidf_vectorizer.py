"""Module containing the TfidfVectorizerCustom class."""

import math
from collections import defaultdict
from collections.abc import Iterable

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from language_classifier.custom_components.vectorizers.vectorizer_custom import (
    VectorizerCustom,
)


class TfidfVectorizerCustom(VectorizerCustom):
    """
    TF-IDF vectorizer implementation extending VectorizerCustom.

    This class computes the term frequency-inverse document frequency representation
    of a collection of documents. It builds the vocabulary and IDF during fitting,
    then transforms input documents into normalized sparse TF-IDF vectors.

    Attributes:
        idf (list[float]): List of inverse document frequency values for each term in the vocabulary.

    """

    def __init__(self) -> None:
        """
        Initialize the TF-IDF vectorizer.

        Calls the superclass initializer and sets up an empty IDF list.
        """
        super().__init__()
        self.idf = []

    def fit_transform(self, input_phrases: Iterable[str]) -> csr_matrix:
        """
        Learn the vocabulary and IDF from the input and return TF-IDF vectors.

        Processes the input documents to build the vocabulary and compute document frequencies,
        then calculates IDF values and constructs normalized TF-IDF sparse vectors for every phrase.

        Args:
            input_phrases (Iterable[str]): Iterable of text documents to fit and transform.

        Returns:
            csr_matrix: Normalized TF-IDF sparse matrix representation of the phrases.

        """
        n_docs = len(input_phrases)
        doc_freq = defaultdict(int)
        vocab = defaultdict(lambda: len(vocab))

        # First pass: build vocabulary and document frequency counts
        tokenized_docs = []
        for doc in input_phrases:
            tokens = self._tokenize(doc)
            token_ids = []
            seen = set()
            for token in tokens:
                idx = vocab[token]
                token_ids.append(idx)
                if idx not in seen:
                    doc_freq[idx] += 1
                    seen.add(idx)
            tokenized_docs.append(token_ids)

        self.vocab = dict(vocab)
        self.vocab_size = len(self.vocab)
        self.idf = [
            math.log((1 + n_docs) / (1 + doc_freq[i])) + 1
            for i in range(self.vocab_size)
        ]

        # Second pass: compute TF-IDF matrix
        rows, cols, data = [], [], []
        for i, token_ids in enumerate(tokenized_docs):
            tf = defaultdict(int)
            for idx in token_ids:
                tf[idx] += 1
            max_tf = max(tf.values())
            for idx, freq in tf.items():
                data.append((freq / max_tf) * self.idf[idx])
                rows.append(i)
                cols.append(idx)
        X = csr_matrix((data, (rows, cols)), shape=(n_docs, self.vocab_size))
        return normalize(X, norm="l2", axis=1)

    def transform(self, input_phrases: Iterable[str]) -> csr_matrix:
        """
        Transform input documents to normalized TF-IDF vectors using the learned vocabulary and IDF.

        Converts new documents into TF-IDF sparse matrix representation, using the vocabulary
        and IDF values computed during fitting.

        Args:
            input_phrases (Iterable[str]): Iterable of text documents to transform.

        Returns:
            csr_matrix: Normalized TF-IDF sparse matrix of the input documents.

        """
        n_docs = len(input_phrases)
        rows, cols, data = [], [], []

        for i, doc in enumerate(input_phrases):
            tf = defaultdict(int)
            tokens = self._tokenize(doc)
            for token in tokens:
                if token in self.vocab:
                    idx = self.vocab[token]
                    tf[idx] += 1
            if not tf:
                continue
            max_tf = max(tf.values())
            for idx, freq in tf.items():
                data.append((freq / max_tf) * self.idf[idx])
                rows.append(i)
                cols.append(idx)

        X = csr_matrix((data, (rows, cols)), shape=(n_docs, self.vocab_size))
        return normalize(X, norm="l2", axis=1)
