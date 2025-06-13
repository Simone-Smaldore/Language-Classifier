import numpy as np
from scipy.sparse import csr_matrix
from numpy.typing import NDArray
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from language_classifier.custom_components.models.model_custom import ModelCustom


class MultinomialNBCustom(ModelCustom, ClassifierMixin, BaseEstimator):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.classes_ = None

    def fit(self, X: csr_matrix, y: NDArray[np.float64]) -> None:
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Inizializza matrici
        class_count = np.zeros(n_classes)
        feature_count = np.zeros((n_classes, n_features))

        y = np.array(y)
        for idx, c in enumerate(self.classes_):
            # Seleziona i documenti della classe c
            X_c = X[y == c]
            class_count[idx] = X_c.shape[0]
            feature_count[idx, :] = X_c.sum(axis=0)

        # Calcolo della log prior
        self.class_log_prior_ = np.log(class_count / n_samples)

        # Calcolo della probabilità condizionata P(w | c)
        smoothed_fc = feature_count + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1).reshape(-1, 1)
        self.feature_log_prob_ = np.log(smoothed_fc / smoothed_cc)

    def predict(self, X: csr_matrix) -> NDArray[np.int32]:
        return self.classes_[np.argmax(self._predict_log_proba(X), axis=1)]

    def _predict_log_proba(self, X: csr_matrix) -> NDArray[np.float64]:
        return X.dot(self.feature_log_prob_.T) + self.class_log_prior_

    def predict_proba(self, X: csr_matrix) -> NDArray[np.float64]:
        log_probs = self._predict_log_proba(X)
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        return probs / probs.sum(axis=1, keepdims=True)
