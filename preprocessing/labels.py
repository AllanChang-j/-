from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd


TaskType = Literal["binary", "three_class", "regression"]


def add_prediction_labels(
    df: pd.DataFrame,
    horizon: int,
    task: TaskType = "binary",
    price_column: str = "adjusted_close",
    neutral_threshold: float = 0.01,
    regression_target: str = "future_return",
) -> pd.DataFrame:
    labeled = df.sort_values(["symbol", "date"]).copy()
    grouped = labeled.groupby("symbol")
    future_price = grouped[price_column].shift(-horizon)
    labeled["future_return"] = future_price / labeled[price_column] - 1
    labeled["signal_date"] = labeled["date"]
    labeled["entry_date"] = grouped["date"].shift(-1)
    labeled["exit_date"] = grouped["date"].shift(-(horizon + 1))
    labeled["entry_price"] = grouped["open"].shift(-1)
    labeled["exit_price"] = grouped["open"].shift(-(horizon + 1))
    labeled["execution_return"] = labeled["exit_price"] / labeled["entry_price"] - 1

    if task == "binary":
        labeled["target"] = (labeled["future_return"] > 0).astype(float)
    elif task == "three_class":
        labeled["target"] = np.select(
            [labeled["future_return"] < -neutral_threshold, labeled["future_return"] > neutral_threshold],
            [0, 2],
            default=1,
        ).astype(float)
    elif task == "regression":
        if regression_target == "daily_future_return_percentile":
            labeled["target"] = labeled.groupby("date")["future_return"].rank(method="average", pct=True).astype(float)
        else:
            labeled["target"] = labeled["future_return"].astype(float)
    else:
        raise ValueError(f"Unsupported task: {task}")

    return labeled.loc[labeled["future_return"].notna() & labeled["target"].notna()]


def num_classes(task: TaskType) -> int:
    if task == "binary":
        return 2
    if task == "three_class":
        return 3
    if task == "regression":
        return 1
    raise ValueError(f"Unsupported task: {task}")
