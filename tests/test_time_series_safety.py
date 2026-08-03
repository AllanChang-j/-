from __future__ import annotations

import pandas as pd

from preprocessing.windows import make_sliding_windows, split_frame_by_dates
from validation.splitters import walk_forward_splits


def test_sliding_windows_do_not_cross_symbols() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 2),
            "symbol": ["A", "A", "A", "B", "B", "B"],
            "name": ["A", "A", "A", "B", "B", "B"],
            "market": ["T"] * 6,
            "feature": [1, 2, 3, 100, 200, 300],
            "target": [0, 1, 0, 1, 0, 1],
            "adjusted_close": [1, 2, 3, 10, 20, 30],
            "future_return": [0.1] * 6,
        }
    )
    dataset = make_sliding_windows(df, ["feature"], sequence_length=2)
    windows = dataset.X[:, :, 0].tolist()
    assert [1, 2] in windows
    assert [2, 3] in windows
    assert [100, 200] in windows
    assert [200, 300] in windows
    assert [3, 100] not in windows


def test_walk_forward_split_is_by_date_with_purge() -> None:
    dates = pd.bdate_range("2024-01-01", periods=20)
    split = next(walk_forward_splits(dates, n_splits=1, train_window=10, validation_window=5, step_size=5, purge=3))
    assert len(set(split.train_dates).intersection(set(split.validation_dates))) == 0
    assert len(split.purge_dates) == 3
    assert max(split.train_dates) < min(split.purge_dates)
    assert max(split.purge_dates) < min(split.validation_dates)


def test_holdout_split_leaves_purge_gap_between_sets() -> None:
    dates = pd.bdate_range("2024-01-01", periods=30)
    df = pd.DataFrame({"date": dates, "symbol": ["A"] * len(dates), "target": [0] * len(dates)})
    train_df, validation_df, test_df = split_frame_by_dates(df, validation_size=0.2, test_size=0.2, purge=3)
    train_dates = sorted(pd.to_datetime(train_df["date"]).unique())
    validation_dates = sorted(pd.to_datetime(validation_df["date"]).unique())
    test_dates = sorted(pd.to_datetime(test_df["date"]).unique())
    assert train_dates[-1] < validation_dates[0]
    assert validation_dates[-1] < test_dates[0]
    assert len(pd.bdate_range(train_dates[-1], validation_dates[0])) - 2 >= 3
    assert len(pd.bdate_range(validation_dates[-1], test_dates[0])) - 2 >= 3
