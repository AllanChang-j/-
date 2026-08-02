from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PredictionResult:
    model_name: str
    predictions: np.ndarray
    probabilities: np.ndarray | None
    targets: np.ndarray
    meta: pd.DataFrame
    metrics: dict[str, Any] = field(default_factory=dict)
    training_time_sec: float = 0.0
    inference_time_sec: float = 0.0
    model_size_bytes: int = 0
    history: dict[str, list[float]] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

