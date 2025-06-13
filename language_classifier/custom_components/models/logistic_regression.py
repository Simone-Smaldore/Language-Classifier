import numpy as np
from scipy.sparse import csr_matrix
from numpy.typing import NDArray
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from language_classifier.custom_components.models.model_custom import ModelCustom


class LogisticRegressionCustom(ModelCustom, ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        learning_rate=0.1,
        epochs=1000,
        threshold=0.5,
        lambda_coeff=1e-6,
        gradient_tolerance=1e-4,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold
        self.lambda_coeff = lambda_coeff
        self.gradient_tolerance = gradient_tolerance
        self.w = None
        self.b = 0

    def fit(self, X: csr_matrix, y: NDArray[np.float64]) -> None:
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
                    f"Early stopping at epoch {epoch+1}: dw norm {np.linalg.norm(dw):.6f} < tol {self.gradient_tolerance}"
                )
                break
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X: csr_matrix) -> NDArray[np.int32]:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def predict_proba(self, X: csr_matrix) -> NDArray[np.float64]:
        z = X.dot(self.w) + self.b
        return self._sigmoid(z)

    def _sigmoid(self, z) -> NDArray[np.float64]:
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
