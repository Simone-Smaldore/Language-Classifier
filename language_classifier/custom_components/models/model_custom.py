import numpy as np
from scipy.sparse import csr_matrix
from abc import ABC, abstractmethod
from numpy.typing import NDArray


class ModelCustom(ABC):

    @abstractmethod
    def fit(self, X: csr_matrix, y: NDArray[np.float64]) -> None:
        pass

    @abstractmethod
    def predict(self, X: csr_matrix) -> NDArray[np.int32]:
        pass
