"""Module implementing a custom Logistic Regression classifier."""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin

from language_classifier.custom_components.models.model_custom import ModelCustom


class LogisticRegressionCustom(ModelCustom, ClassifierMixin, BaseEstimator):
    """
    Custom implementation of logistic regression classifier.

    This class implements a binary logistic regression model with L2 regularization,
    trained using gradient descent with early stopping based on the norm of the gradient.

    Attributes:
        learning_rate (float): Step size for gradient descent updates.
        epochs (int): Maximum number of training iterations.
        threshold (float): Decision threshold for classifying samples.
        lambda_coeff (float): Regularization coefficient for L2 penalty.
        gradient_tolerance (float): Threshold for the gradient norm to trigger early stopping.
        w (NDArray): Weight vector learned during training.
        b (float): Bias term learned during training.
        classes_ (NDArray): Array of unique class labels seen during training.

    Methods:
        fit(X, y): Train the logistic regression model.
        predict(X): Predict binary class labels for samples.
        predict_proba(X): Predict probabilities for the positive class.
        _sigmoid(z): Compute the sigmoid activation function.

    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        epochs: int = 1000,
        threshold: float = 0.5,
        lambda_coeff: float = 1e-6,
        gradient_tolerance: float = 1e-4,
    ) -> None:
        """
        Initialize the LogisticRegressionCustom instance with training parameters.

        Args:
            learning_rate (float, optional): Gradient descent step size. Default is 0.1.
            epochs (int, optional): Maximum number of iterations to run during training. Default is 1000.
            threshold (float, optional): Threshold for converting probabilities to binary labels. Default is 0.5.
            lambda_coeff (float, optional): L2 regularization strength. Default is 1e-6.
            gradient_tolerance (float, optional): Gradient norm threshold for early stopping. Default is 1e-4.

        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold
        self.lambda_coeff = lambda_coeff
        self.gradient_tolerance = gradient_tolerance
        self.b = 0

    def fit(self, X: csr_matrix, y: NDArray[np.float64]) -> None:
        """
        Train the logistic regression model using gradient descent.

        Args:
            X (csr_matrix): Sparse feature matrix of shape (n_samples, n_features).
            y (NDArray[np.float64]): Target binary labels array of shape (n_samples,).

        Notes:
            - Uses L2 regularization with coefficient `lambda_coeff`.
            - Implements early stopping if the norm of the weight gradient falls below `gradient_tolerance`.

        """
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        for epoch in range(self.epochs):
            z = X.dot(self.w) + self.b
            y_pred = self._sigmoid(z)
            error = y_pred - y
            dw = (X.T.dot(error) / n_samples) + self.lambda_coeff * self.w
            db = np.sum(error) / n_samples
            if np.linalg.norm(dw) < self.gradient_tolerance:
                print(
                    f"Early stopping at epoch {epoch+1}: dw norm {np.linalg.norm(dw):.6f} < tol {self.gradient_tolerance}",
                )
                break
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X: csr_matrix) -> NDArray[np.int32]:
        """
        Predict binary class labels for the input samples.

        Args:
            X (csr_matrix): Sparse feature matrix of shape (n_samples, n_features).

        Returns:
            NDArray[np.int32]: Predicted class labels (0 or 1) of shape (n_samples,).

        """
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def predict_proba(self, X: csr_matrix) -> NDArray[np.float64]:
        """
        Predict probabilities  for the input samples.

        Args:
            X (csr_matrix): Sparse feature matrix of shape (n_samples, n_features).

        Returns:
            NDArray[np.float64]: Predicted probabilities of shape (n_samples,).

        """
        z = X.dot(self.w) + self.b
        return self._sigmoid(z)

    def _sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the sigmoid activation function.

        Args:
            z (array-like): Input array or value.

        Returns:
            NDArray[np.float64]: Sigmoid output in the range (0, 1).

        Notes:
            - Clips input `z` to range [-500, 500] for numerical stability.

        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
