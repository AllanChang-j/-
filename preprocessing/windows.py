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

    def flatten_feature_names(self) -> list[str]:
        names: list[str] = []
        sequence_length = self.X.shape[1]
        for step in range(sequence_length):
            lag = sequence_length - 1 - step
            suffix = "t0" if lag == 0 else f"t-{lag}"
            names.extend(f"{feature}_{suffix}" for feature in self.feature_columns)
        return names


def make_sliding_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
    target_column: str = "target",
    price_column: str = "adjusted_close",
    target_dates: set[pd.Timestamp] | None = None,
) -> WindowDataset:
    X_parts: list[np.ndarray] = []
    y_parts: list[float] = []
    meta_records: list[dict[str, object]] = []

    for symbol, group in df.sort_values(["symbol", "date"]).groupby("symbol"):
        group = group.reset_index(drop=True)
        feature_values = group[feature_columns].to_numpy(dtype=float)
        targets = group[target_column].to_numpy(dtype=float)
        for end in range(sequence_length - 1, len(group)):
            row = group.iloc[end]
            row_date = pd.Timestamp(row["date"])
            if target_dates is not None and row_date not in target_dates:
                continue
            window = feature_values[end - sequence_length + 1 : end + 1]
            target = targets[end]
            if np.isnan(window).any() or np.isnan(target):
                continue
            X_parts.append(window)
            y_parts.append(target)
            meta_records.append(
                {
                    "date": row["date"],
                    "symbol": symbol,
                    "name": row.get("name", symbol),
                    "market": row.get("market", "unknown"),
                    "close": row.get(price_column, row.get("close")),
                    "future_return": row.get("future_return"),
                    "execution_return": row.get("execution_return", row.get("future_return")),
                    "signal_date": row.get("signal_date", row["date"]),
                    "entry_date": row.get("entry_date"),
                    "exit_date": row.get("exit_date"),
                    "entry_price": row.get("entry_price"),
                    "exit_price": row.get("exit_price"),
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
    purge: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(pd.to_datetime(df["date"]).unique()))
    if len(dates) < 10:
        raise ValueError("Not enough unique dates to create train/validation/test splits.")
    test_count = max(1, int(len(dates) * test_size))
    validation_count = max(1, int(len(dates) * validation_size))
    train_end = len(dates) - validation_count - test_count
    if train_end <= 0:
        raise ValueError("validation_size and test_size leave no training dates.")
    validation_end = train_end + validation_count
    train_dates = set(dates[: max(0, train_end - purge)])
    validation_dates = set(dates[train_end : max(train_end, validation_end - purge)])
    test_dates = set(dates[train_end + validation_count :])
    train_df = df[df["date"].isin(train_dates)].copy()
    validation_df = df[df["date"].isin(validation_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()
    return train_df, validation_df, test_df
