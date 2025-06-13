import numpy as np
from scipy.sparse import csr_matrix
from abc import ABC, abstractmethod
from numpy.typing import NDArray


from sklearn.model_selection import KFold
from typing import Type, Dict, Any, List, Tuple
import numpy as np
from itertools import product


class ModelCustom(ABC):

    @abstractmethod
    def fit(self, X: csr_matrix, y: NDArray[np.float64]) -> None:
        pass

    @abstractmethod
    def predict(self, X: csr_matrix) -> NDArray[np.int32]:
        pass


class GridSearchCVCustom:
    def __init__(
        self,
        model_class: Type[ModelCustom],
        param_grid: dict[str, list[str]],
        cv: int = 5,
        scoring: str = "accuracy",
    ):
        self.model_class = model_class
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.best_model: ModelCustom = None
        self.best_score = -np.inf
        self.best_params = {}

    def _score(self, y_true, y_pred) -> float:
        if self.scoring == "accuracy":
            return np.mean(y_true == y_pred)
        raise ValueError(f"Scoring '{self.scoring}' not supported.")

    def _param_combinations(self) -> List[Dict[str, Any]]:
        keys, values = zip(*self.param_grid.items())
        cartesian_product = product(*values)
        return [dict(zip(keys, v)) for v in cartesian_product]

    def fit(self, X: csr_matrix, y: NDArray[np.float64]) -> None:
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=1999)
        y = np.array(y)
        for params in self._param_combinations():
            print(f"Trying params {params}")
            scores = []

            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = self.model_class(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = self._score(y_val, y_pred)
                scores.append(score)

            avg_score = np.mean(scores)

            if avg_score > self.best_score:
                self.best_score = avg_score
                self.best_params = params
                self.best_model = self.model_class(**params)
                self.best_model.fit(X, y)
        print(f"Best params: {self.best_params}")

    def predict(self, X: csr_matrix) -> NDArray[np.int32]:
        return self.best_model.predict(X)


class LogisticRegressionCustom(ModelCustom):
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


class MultinomialNBCustom(ModelCustom):
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
