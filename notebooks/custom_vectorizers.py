import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections import defaultdict
import math
import re
from typing import Iterable
from abc import ABC, abstractmethod
from sklearn.preprocessing import normalize


class VectorizerCustom(ABC):

    def __init__(self):
        self.vocab = {}
        self.vocab_size = 0

    @abstractmethod
    def fit_transform(self, corpus: Iterable[str]) -> csr_matrix:
        pass

    @abstractmethod
    def transform(self, corpus: Iterable[str]) -> csr_matrix:
        pass

    def get_feature_names_out(self) -> list[str]:
        sorted_vocab = sorted(self.vocab.items(), key=lambda x: x[1])
        return [word for word, idx in sorted_vocab]

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text)


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
