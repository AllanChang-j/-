from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


@dataclass(frozen=True)
class TimeSplit:
    name: str
    train_dates: np.ndarray
    validation_dates: np.ndarray


def walk_forward_splits(
    dates: list[pd.Timestamp] | np.ndarray,
    n_splits: int,
    train_window: int,
    validation_window: int,
    step_size: int,
    expanding: bool = True,
) -> Iterator[TimeSplit]:
    unique_dates = np.array(sorted(pd.to_datetime(dates).unique()))
    max_start = len(unique_dates) - validation_window
    split_id = 0
    train_start = 0
    train_end = min(train_window, max_start)
    while split_id < n_splits and train_end + validation_window <= len(unique_dates):
        validation_start = train_end
        validation_end = validation_start + validation_window
        if expanding:
            train_dates = unique_dates[:train_end]
        else:
            train_dates = unique_dates[train_start:train_end]
        validation_dates = unique_dates[validation_start:validation_end]
        if len(train_dates) and len(validation_dates):
            yield TimeSplit(f"walk_forward_{split_id + 1}", train_dates, validation_dates)
        split_id += 1
        train_end += step_size
        if not expanding:
            train_start += step_size


def rolling_window_splits(
    dates: list[pd.Timestamp] | np.ndarray,
    n_splits: int,
    train_window: int,
    validation_window: int,
    step_size: int,
) -> Iterator[TimeSplit]:
    return walk_forward_splits(
        dates=dates,
        n_splits=n_splits,
        train_window=train_window,
        validation_window=validation_window,
        step_size=step_size,
        expanding=False,
    )


def sklearn_time_series_splits(dates: list[pd.Timestamp] | np.ndarray, n_splits: int) -> Iterator[TimeSplit]:
    unique_dates = np.array(sorted(pd.to_datetime(dates).unique()))
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for index, (train_idx, validation_idx) in enumerate(splitter.split(unique_dates), start=1):
        yield TimeSplit(f"time_series_split_{index}", unique_dates[train_idx], unique_dates[validation_idx])


def apply_date_split(df: pd.DataFrame, split: TimeSplit) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_set = set(split.train_dates)
    validation_set = set(split.validation_dates)
    train_df = df[pd.to_datetime(df["date"]).isin(train_set)].copy()
    validation_df = df[pd.to_datetime(df["date"]).isin(validation_set)].copy()
    return train_df, validation_df
