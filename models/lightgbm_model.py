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

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_valid: np.ndarray | None = None,
        y_valid: np.ndarray | None = None,
    ) -> "LightGBMWrapper":
        params = dict(self.params)
        early_stopping_rounds = params.pop("early_stopping_rounds", None)
        log_evaluation_period = int(params.pop("log_evaluation_period", 50))
        try:
            import lightgbm as lgb

            if self.task == "regression":
                self.model = lgb.LGBMRegressor(**params)
            else:
                objective = "binary" if len(np.unique(y)) <= 2 else "multiclass"
                self.model = lgb.LGBMClassifier(objective=objective, **params)
            self.backend = "lightgbm"
            fit_kwargs: dict[str, Any] = {}
            callbacks = [lgb.log_evaluation(period=log_evaluation_period)]
            if X_valid is not None and y_valid is not None:
                fit_kwargs["eval_set"] = [(X_valid, y_valid)]
                if early_stopping_rounds:
                    callbacks.append(lgb.early_stopping(stopping_rounds=int(early_stopping_rounds)))
            fit_kwargs["callbacks"] = callbacks
            self.model.fit(X, y, **fit_kwargs)
        except Exception:
            from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

            if self.task == "regression":
                self.model = HistGradientBoostingRegressor(
                    max_iter=int(params.get("n_estimators", 300)),
                    learning_rate=float(params.get("learning_rate", 0.03)),
                    l2_regularization=float(params.get("reg_lambda", 0.1)),
                    random_state=int(params.get("random_state", 42)),
                )
            else:
                self.model = HistGradientBoostingClassifier(
                    max_iter=int(params.get("n_estimators", 300)),
                    learning_rate=float(params.get("learning_rate", 0.03)),
                    l2_regularization=float(params.get("reg_lambda", 0.1)),
                    random_state=int(params.get("random_state", 42)),
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
