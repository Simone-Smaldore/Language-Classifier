"""Module implementing a custom Multinomial Naive Bayes classifier."""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin

from language_classifier.custom_components.models.model_custom import ModelCustom


class MultinomialNBCustom(ModelCustom, ClassifierMixin, BaseEstimator):
    """
    Custom Multinomial Naive Bayes classifier.

    Implements a multinomial Naive Bayes model for classification tasks on sparse input data.
    Supports additive smoothing controlled by the alpha parameter.

    Attributes:
        alpha (float): Additive smoothing parameter (default 1.0).
        log_prob_prior (NDArray[np.float64]): Log prior probabilities of classes.
        log_prob_cond (NDArray[np.float64]): Log probability of features given classes.
        classes_ (NDArray): Array of class labels.

    """

    def __init__(self, alpha: float = 1.0) -> None:
        """
        Initialize the MultinomialNBCustom classifier.

        Args:
            alpha (float, optional): Additive smoothing parameter. Defaults to 1.0.

        """
        self.alpha = alpha
        self.log_prob_prior = None
        self.log_prob_cond = None
        self.classes_ = None

    def fit(self, X: csr_matrix, y: NDArray[np.float64]) -> None:
        """
        Fit the Naive Bayes classifier according to the given training data.

        Computes class prior probabilities and conditional feature probabilities
        with additive smoothing.

        Args:
            X (csr_matrix): Sparse matrix of shape (n_samples, n_features) with term counts.
            y (NDArray[np.float64]): Array of target class labels.

        Returns:
            None

        """
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
        self.log_prob_prior = np.log(class_count / n_samples)

        # Calcolo della probabilità condizionata P(w | c)
        smoothed_fc = feature_count + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1).reshape(-1, 1)
        self.log_prob_cond = np.log(smoothed_fc / smoothed_cc)

    def predict(self, X: csr_matrix) -> NDArray[np.int32]:
        """
        Perform classification on an array of test vectors X.

        Args:
            X (csr_matrix): Sparse matrix of shape (n_samples, n_features).

        Returns:
            NDArray[np.int32]: Predicted class labels for samples in X.

        """
        return self.classes_[np.argmax(self._predict_log_proba(X), axis=1)]

    def _predict_log_proba(self, X: csr_matrix) -> NDArray[np.float64]:
        """
        Calculate the log-probability of each class for the input samples.

        Args:
            X (csr_matrix): Sparse matrix of shape (n_samples, n_features).

        Returns:
            NDArray[np.float64]: Log probabilities of shape (n_samples, n_classes).

        """
        return X.dot(self.log_prob_cond.T) + self.log_prob_prior

    def predict_proba(self, X: csr_matrix) -> NDArray[np.float64]:
        """
        Return probability estimates for the test vectors X.

        Applies the softmax function to log-probabilities to obtain probabilities.

        Args:
            X (csr_matrix): Sparse matrix of shape (n_samples, n_features).

        Returns:
            NDArray[np.float64]: Probability estimates of shape (n_samples, n_classes).

        """
        log_probs = self._predict_log_proba(X)
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        return probs / probs.sum(axis=1, keepdims=True)
