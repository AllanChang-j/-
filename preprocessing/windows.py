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
    meta_frames: list[pd.DataFrame] = []

    for symbol, group in df.sort_values(["symbol", "date"]).groupby("symbol"):
        group = group.reset_index(drop=True)
        feature_values = group[feature_columns].to_numpy(dtype=float)
        targets = group[target_column].to_numpy(dtype=float)
        row_complete = np.isfinite(feature_values).all(axis=1)
        window_complete = pd.Series(row_complete).rolling(sequence_length).sum().to_numpy() == sequence_length
        target_complete = np.isfinite(targets)
        if target_dates is None:
            date_in_scope = np.ones(len(group), dtype=bool)
        else:
            date_in_scope = group["date"].map(lambda value: pd.Timestamp(value) in target_dates).to_numpy(dtype=bool)
        valid_ends = np.flatnonzero(window_complete & target_complete & date_in_scope)
        valid_ends = valid_ends[valid_ends >= sequence_length - 1]
        if len(valid_ends) == 0:
            continue

        X_parts.extend(feature_values[end - sequence_length + 1 : end + 1] for end in valid_ends)
        y_parts.extend(targets[valid_ends].tolist())
        selected = group.iloc[valid_ends]
        fallback_close = selected["close"] if "close" in selected else np.nan
        meta_frames.append(
            pd.DataFrame(
                {
                    "date": selected["date"].to_numpy(),
                    "symbol": symbol,
                    "name": selected["name"].to_numpy() if "name" in selected else symbol,
                    "market": selected["market"].to_numpy() if "market" in selected else "unknown",
                    "close": selected[price_column].to_numpy() if price_column in selected else fallback_close,
                    "future_return": selected["future_return"].to_numpy() if "future_return" in selected else np.nan,
                    "execution_return": selected["execution_return"].to_numpy()
                    if "execution_return" in selected
                    else selected["future_return"].to_numpy()
                    if "future_return" in selected
                    else np.nan,
                    "signal_date": selected["signal_date"].to_numpy() if "signal_date" in selected else selected["date"].to_numpy(),
                    "entry_date": selected["entry_date"].to_numpy() if "entry_date" in selected else np.nan,
                    "exit_date": selected["exit_date"].to_numpy() if "exit_date" in selected else np.nan,
                    "entry_price": selected["entry_price"].to_numpy() if "entry_price" in selected else np.nan,
                    "exit_price": selected["exit_price"].to_numpy() if "exit_price" in selected else np.nan,
                }
            )
        )

    if not X_parts:
        raise ValueError("No sliding windows were created. Check sequence_length, missing values, and data length.")

    return WindowDataset(
        X=np.asarray(X_parts, dtype=np.float32),
        y=np.asarray(y_parts, dtype=np.float32),
        meta=pd.concat(meta_frames, ignore_index=True),
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
