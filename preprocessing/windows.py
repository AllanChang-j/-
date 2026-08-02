from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowDataset:
    X: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame
    feature_columns: list[str]

    def flatten(self) -> np.ndarray:
        return self.X.reshape(self.X.shape[0], self.X.shape[1] * self.X.shape[2])


def make_sliding_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
    target_column: str = "target",
    price_column: str = "adjusted_close",
) -> WindowDataset:
    X_parts: list[np.ndarray] = []
    y_parts: list[float] = []
    meta_records: list[dict[str, object]] = []

    for symbol, group in df.sort_values(["symbol", "date"]).groupby("symbol"):
        group = group.reset_index(drop=True)
        feature_values = group[feature_columns].to_numpy(dtype=float)
        targets = group[target_column].to_numpy(dtype=float)
        for end in range(sequence_length - 1, len(group)):
            window = feature_values[end - sequence_length + 1 : end + 1]
            target = targets[end]
            if np.isnan(window).any() or np.isnan(target):
                continue
            X_parts.append(window)
            y_parts.append(target)
            row = group.iloc[end]
            meta_records.append(
                {
                    "date": row["date"],
                    "symbol": symbol,
                    "name": row.get("name", symbol),
                    "market": row.get("market", "unknown"),
                    "close": row.get(price_column, row.get("close")),
                    "future_return": row.get("future_return"),
                }
            )

    if not X_parts:
        raise ValueError("No sliding windows were created. Check sequence_length, missing values, and data length.")

    return WindowDataset(
        X=np.asarray(X_parts, dtype=np.float32),
        y=np.asarray(y_parts, dtype=np.float32),
        meta=pd.DataFrame(meta_records),
        feature_columns=list(feature_columns),
    )


def split_frame_by_dates(
    df: pd.DataFrame,
    validation_size: float,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(pd.to_datetime(df["date"]).unique()))
    if len(dates) < 10:
        raise ValueError("Not enough unique dates to create train/validation/test splits.")
    test_count = max(1, int(len(dates) * test_size))
    validation_count = max(1, int(len(dates) * validation_size))
    train_end = len(dates) - validation_count - test_count
    if train_end <= 0:
        raise ValueError("validation_size and test_size leave no training dates.")
    train_dates = set(dates[:train_end])
    validation_dates = set(dates[train_end : train_end + validation_count])
    test_dates = set(dates[train_end + validation_count :])
    train_df = df[df["date"].isin(train_dates)].copy()
    validation_df = df[df["date"].isin(validation_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()
    return train_df, validation_df, test_df
