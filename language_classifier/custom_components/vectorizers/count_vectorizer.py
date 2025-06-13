from scipy.sparse import csr_matrix
from collections import defaultdict
from typing import Iterable
from language_classifier.custom_components.vectorizers.vectorizer_custom import (
    VectorizerCustom,
)


class CountVectorizerCustom(VectorizerCustom):
    def __init__(self):
        super().__init__()

    def fit_transform(self, corpus: Iterable[str]) -> csr_matrix:
        rows, cols, data = [], [], []
        vocab = defaultdict(lambda: len(vocab))
        for i, doc in enumerate(corpus):
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
        return csr_matrix((data, (rows, cols)), shape=(len(corpus), self.vocab_size))

    def transform(self, corpus: Iterable[str]) -> csr_matrix:
        rows, cols, data = [], [], []

        for i, doc in enumerate(corpus):
            word_counts = defaultdict(int)
            for token in self._tokenize(doc):
                if token in self.vocab:
                    word_id = self.vocab[token]
                    word_counts[word_id] += 1
            for word_id, count in word_counts.items():
                data.append(count)
                rows.append(i)
                cols.append(word_id)

        return csr_matrix((data, (rows, cols)), shape=(len(corpus), self.vocab_size))
