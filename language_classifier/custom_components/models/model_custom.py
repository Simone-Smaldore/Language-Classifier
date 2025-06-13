"""The module defines the abstract base class ModelCustom."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix


class ModelCustom(ABC):
    """
    Abstract base class for custom machine learning models.

    This class defines the interface that all derived models must implement,
    requiring fit and predict methods compatible with sparse input matrices.

    Methods:
        fit(X, y): Train the model using input features and target labels.
        predict(X): Predict target labels for given input features.

    """

    @abstractmethod
    def fit(self, X: csr_matrix, y: NDArray[np.float64]) -> None:
        """
        Train the model with input data and target labels.

        Args:
            X (csr_matrix): Sparse feature matrix of shape (n_samples, n_features).
            y (NDArray[np.float64]): Target labels array of shape (n_samples,).

        """

    @abstractmethod
    def predict(self, X: csr_matrix) -> NDArray[np.int32]:
        """
        Predict target labels for given input features.

        Args:
            X (csr_matrix): Sparse feature matrix of shape (n_samples, n_features).

        Returns:
            NDArray[np.int32]: Predicted labels array of shape (n_samples,).

        """
