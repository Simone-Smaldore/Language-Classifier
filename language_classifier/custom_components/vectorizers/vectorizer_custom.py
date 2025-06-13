from scipy.sparse import csr_matrix
import re
from typing import Iterable
from abc import ABC, abstractmethod


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
