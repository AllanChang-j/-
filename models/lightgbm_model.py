from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np


@dataclass
class LightGBMWrapper:
    task: str
    params: dict[str, Any]
    model: Any | None = None
    backend: str = "lightgbm"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMWrapper":
        try:
            import lightgbm as lgb

            if self.task == "regression":
                self.model = lgb.LGBMRegressor(**self.params)
            else:
                objective = "binary" if len(np.unique(y)) <= 2 else "multiclass"
                self.model = lgb.LGBMClassifier(objective=objective, **self.params)
            self.backend = "lightgbm"
        except Exception:
            from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

            if self.task == "regression":
                self.model = HistGradientBoostingRegressor(
                    max_iter=int(self.params.get("n_estimators", 300)),
                    learning_rate=float(self.params.get("learning_rate", 0.03)),
                    l2_regularization=float(self.params.get("reg_lambda", 0.1)),
                    random_state=int(self.params.get("random_state", 42)),
                )
            else:
                self.model = HistGradientBoostingClassifier(
                    max_iter=int(self.params.get("n_estimators", 300)),
                    learning_rate=float(self.params.get("learning_rate", 0.03)),
                    l2_regularization=float(self.params.get("reg_lambda", 0.1)),
                    random_state=int(self.params.get("random_state", 42)),
                )
            self.backend = "sklearn_hist_gradient_boosting_fallback"
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if self.model is None or self.task == "regression":
            return None
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        predictions = self.model.predict(X)
        classes = np.unique(predictions).astype(int)
        proba = np.zeros((len(predictions), max(classes.max() + 1, 2)))
        proba[np.arange(len(predictions)), predictions.astype(int)] = 1.0
        return proba

    def save(self, path: str) -> None:
        joblib.dump({"backend": self.backend, "task": self.task, "params": self.params, "model": self.model}, path)
