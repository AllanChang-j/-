from __future__ import annotations

from typing import Any

import pandas as pd

from feature_engineering.technical import build_technical_features, candidate_feature_columns


def build_feature_frame(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    windows = [int(value) for value in config.get("features", {}).get("windows", [5, 10, 20, 60])]
    lag_windows = [int(value) for value in config.get("features", {}).get("lag_windows", [1, 2, 3, 5, 10, 20])]
    features = build_technical_features(df, windows=windows, lag_windows=lag_windows)
    feature_columns = candidate_feature_columns(features)
    return features, feature_columns
