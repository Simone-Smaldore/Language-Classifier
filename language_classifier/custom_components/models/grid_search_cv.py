import numpy as np
from scipy.sparse import csr_matrix
from numpy.typing import NDArray


from sklearn.model_selection import KFold
from typing import Type, Dict, Any, List
import numpy as np
from itertools import product

from language_classifier.custom_components.models.model_custom import ModelCustom


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
        return self.best_model.predict(X)
