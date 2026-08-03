from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureSelectionResult:
    selected_features: list[str]
    ranking: pd.DataFrame
    removed_low_quality: list[str]
    removed_correlated: list[str]


def _drop_low_quality(df: pd.DataFrame, feature_columns: list[str], min_non_null_ratio: float) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    removed: list[str] = []
    for column in feature_columns:
        ratio = df[column].notna().mean()
        variance = df[column].var(skipna=True)
        if ratio < min_non_null_ratio or pd.isna(variance) or variance <= 0:
            removed.append(column)
        else:
            kept.append(column)
    return kept, removed


def _drop_correlated(df: pd.DataFrame, feature_columns: list[str], threshold: float, max_rows: int = 50_000) -> tuple[list[str], list[str]]:
    if len(feature_columns) <= 1:
        return feature_columns, []
    feature_frame = df[feature_columns]
    if len(feature_frame) > max_rows:
        positions = np.linspace(0, len(feature_frame) - 1, max_rows, dtype=int)
        feature_frame = feature_frame.iloc[positions]
    corr = feature_frame.corr(method="pearson").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    removed = [column for column in upper.columns if any(upper[column] > threshold)]
    kept = [column for column in feature_columns if column not in removed]
    return kept, removed


def _rank_from_scores(features: list[str], scores: np.ndarray, name: str) -> pd.DataFrame:
    clean_scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    ranks = pd.Series(clean_scores, index=features).rank(ascending=False, method="average")
    return pd.DataFrame({"feature": features, f"{name}_score": clean_scores, f"{name}_rank": ranks.values})


def select_features(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    task: str,
    target_column: str = "target",
    max_features: int = 120,
    min_non_null_ratio: float = 0.85,
    correlation_threshold: float = 0.96,
    random_state: int = 42,
) -> FeatureSelectionResult:
    kept, removed_low_quality = _drop_low_quality(train_df, feature_columns, min_non_null_ratio)
    kept, removed_correlated = _drop_correlated(train_df, kept, correlation_threshold)
    model_df = train_df[kept + [target_column]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(model_df) < 30:
        selected = kept[:max_features]
        ranking = pd.DataFrame({"feature": selected, "aggregate_rank": np.arange(1, len(selected) + 1)})
        return FeatureSelectionResult(selected, ranking, removed_low_quality, removed_correlated)
    if len(model_df) > 50_000:
        model_df = model_df.sample(n=50_000, random_state=random_state)

    X = model_df[kept].to_numpy(dtype=float)
    y = model_df[target_column].to_numpy()
    is_regression = task == "regression"
    rank_frames: list[pd.DataFrame] = []

    if is_regression:
        lasso = make_pipeline(StandardScaler(), LassoCV(cv=3, random_state=random_state, max_iter=5000))
        elastic = make_pipeline(StandardScaler(), ElasticNetCV(cv=3, random_state=random_state, max_iter=5000))
        ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 25), cv=3))
        lasso.fit(X, y)
        elastic.fit(X, y)
        ridge.fit(X, y)
        rank_frames.append(_rank_from_scores(kept, np.abs(lasso[-1].coef_), "lasso"))
        rank_frames.append(_rank_from_scores(kept, np.abs(elastic[-1].coef_), "elastic_net"))
        rank_frames.append(_rank_from_scores(kept, np.abs(ridge[-1].coef_), "ridge"))
        mi = mutual_info_regression(X, y, random_state=random_state)
        aux_model: Any = RandomForestRegressor(n_estimators=80, min_samples_leaf=5, random_state=random_state, n_jobs=1)
        scoring = "neg_mean_absolute_error"
    else:
        l1 = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l1",
                solver="saga",
                C=0.2,
                random_state=random_state,
                max_iter=800,
            ),
        )
        elastic_lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.5,
                C=0.2,
                random_state=random_state,
                max_iter=800,
            ),
        )
        ridge_lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                C=0.2,
                random_state=random_state,
                max_iter=800,
            ),
        )
        l1.fit(X, y.astype(int))
        elastic_lr.fit(X, y.astype(int))
        ridge_lr.fit(X, y.astype(int))
        l1_coef = np.abs(l1[-1].coef_).mean(axis=0)
        elastic_coef = np.abs(elastic_lr[-1].coef_).mean(axis=0)
        ridge_coef = np.abs(ridge_lr[-1].coef_).mean(axis=0)
        rank_frames.append(_rank_from_scores(kept, l1_coef, "lasso"))
        rank_frames.append(_rank_from_scores(kept, elastic_coef, "elastic_net"))
        rank_frames.append(_rank_from_scores(kept, ridge_coef, "ridge"))
        mi = mutual_info_classif(X, y.astype(int), random_state=random_state)
        aux_model = RandomForestClassifier(
            n_estimators=80,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        )
        scoring = "f1_weighted"

    rank_frames.append(_rank_from_scores(kept, mi, "mutual_info"))
    aux_model.fit(X, y.astype(float) if is_regression else y.astype(int))
    rank_frames.append(_rank_from_scores(kept, aux_model.feature_importances_, "tree"))

    try:
        permutation_size = min(10_000, len(X))
        permutation_X = X[:permutation_size]
        permutation_y = y[:permutation_size]
        perm = permutation_importance(aux_model, permutation_X, permutation_y, n_repeats=2, random_state=random_state, scoring=scoring, n_jobs=1)
        rank_frames.append(_rank_from_scores(kept, perm.importances_mean, "permutation"))
    except Exception:
        pass

    try:
        import shap

        sample_size = min(500, len(X))
        background = X[:sample_size]
        explainer = shap.TreeExplainer(aux_model)
        shap_values = explainer.shap_values(background)
        if isinstance(shap_values, list):
            shap_score = np.mean([np.abs(values).mean(axis=0) for values in shap_values], axis=0)
        else:
            shap_score = np.abs(shap_values).mean(axis=0)
        rank_frames.append(_rank_from_scores(kept, shap_score, "shap"))
    except Exception:
        pass

    ranking = rank_frames[0]
    for frame in rank_frames[1:]:
        ranking = ranking.merge(frame, on="feature", how="outer")

    rank_columns = [column for column in ranking.columns if column.endswith("_rank")]
    ranking["aggregate_rank"] = ranking[rank_columns].mean(axis=1)
    ranking = ranking.sort_values("aggregate_rank").reset_index(drop=True)
    selected_features = ranking["feature"].head(max_features).tolist()
    return FeatureSelectionResult(selected_features, ranking, removed_low_quality, removed_correlated)
