from scipy.sparse import csr_matrix
from collections import defaultdict
import math
from typing import Iterable
from sklearn.preprocessing import normalize
from language_classifier.custom_components.vectorizers.vectorizer_custom import (
    VectorizerCustom,
)


class TfidfVectorizerCustom(VectorizerCustom):
    def __init__(self):
        super().__init__()
        self.idf = []

    def fit_transform(self, corpus: Iterable[str]) -> csr_matrix:
        n_docs = len(corpus)
        doc_freq = defaultdict(int)
        vocab = defaultdict(lambda: len(vocab))

        # Prima passata: costruzione vocabolario e idf
        tokenized_docs = []
        for doc in corpus:
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

        # Seconda passata: calcolo TF-IDF
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

    def transform(self, corpus: Iterable[str]) -> csr_matrix:
        n_docs = len(corpus)
        rows, cols, data = [], [], []

        for i, doc in enumerate(corpus):
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
