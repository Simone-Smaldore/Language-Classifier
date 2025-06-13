"""The module provides the GridSearchCVCustom class."""

from itertools import product
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold

from language_classifier.custom_components.models.model_custom import ModelCustom


class GridSearchCVCustom:
    """
    Custom implementation of grid search with cross-validation.

    This class performs exhaustive search over a parameter grid for a given
    ModelCustom subclass. It uses k-fold cross-validation to evaluate model
    performance and selects the best hyperparameter combination.

    Attributes:
        model_class (type[ModelCustom]): The model class to be tuned.
        param_grid (dict[str, list[str]]): Dictionary defining parameters and their candidate values.
        cv (int): Number of folds for cross-validation.
        scoring (str): Metric used for evaluation ('accuracy' supported).
        best_model (ModelCustom): Best fitted model after grid search.
        best_score (float): Best score achieved during cross-validation.
        best_params (dict): Parameter set corresponding to the best score.

    Methods:
        fit(X, y): Perform grid search with cross-validation and fit best model.
        predict(X): Predict labels using the best fitted model.

    """

    def __init__(
        self,
        model_class: type[ModelCustom],
        param_grid: dict[str, list[str]],
        cv: int = 5,
        scoring: str = "accuracy",
    ) -> None:
        """
        Initialize the GridSearchCVCustom instance.

        Args:
            model_class (type[ModelCustom]): The model class to tune.
            param_grid (dict[str, list[str]]): Hyperparameters and their candidate values.
            cv (int, optional): Number of cross-validation folds. Default is 5.
            scoring (str, optional): Scoring metric for evaluation. Default is 'accuracy'.

        """
        self.model_class = model_class
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.best_model: ModelCustom = None
        self.best_score = -np.inf
        self.best_params = {}

    def _score(self, y_true: NDArray[np.int32], y_pred: NDArray[np.int32]) -> float:
        """
        Compute the evaluation score for predictions.

        Args:
            y_true (NDArray[np.int32]): True target labels.
            y_pred (NDArray[np.int32]): Predicted target labels.

        Returns:
            float: Computed score.

        Raises:
            ValueError: If the scoring metric is not supported.

        """
        if self.scoring == "accuracy":
            return np.mean(y_true == y_pred)
        message = f"Scoring '{self.scoring}' not supported."
        raise ValueError(message)

    def _param_combinations(self) -> list[dict[str, Any]]:
        """
        Generate all combinations of parameters from the parameter grid.

        Returns:
            list[dict[str, Any]]: List of dictionaries representing parameter combinations.

        """
        keys, values = zip(*self.param_grid.items(), strict=False)
        cartesian_product = product(*values)
        return [dict(zip(keys, v, strict=False)) for v in cartesian_product]

    def fit(self, X: csr_matrix, y: NDArray[np.float64]) -> None:
        """
        Perform grid search with cross-validation and fit the best model.

        Args:
            X (csr_matrix): Sparse feature matrix of shape (n_samples, n_features).
            y (NDArray[np.float64]): Target labels array of shape (n_samples,).

        Notes:
            - Uses k-fold cross-validation with shuffling and fixed random state.
            - Updates best_model, best_score, and best_params attributes.
            - Prints best parameters found after fitting.

        """
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=1999)
        y = np.array(y)
        for params in self._param_combinations():
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
        self.best_model = self.model_class(**self.best_params)
        self.best_model.fit(X, y)
        print(f"Best params: {self.best_params}")

    def predict(self, X: csr_matrix) -> NDArray[np.int32]:
        """
        Predict target labels using the best fitted model.

        Args:
            X (csr_matrix): Sparse feature matrix of shape (n_samples, n_features).

        Returns:
            NDArray[np.int32]: Predicted labels array of shape (n_samples,).

        """
        return self.best_model.predict(X)
