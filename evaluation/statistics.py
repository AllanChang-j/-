from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from scipy import stats


def diebold_mariano_test(loss_a: np.ndarray, loss_b: np.ndarray) -> dict[str, float]:
    loss_diff = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    loss_diff = loss_diff[~np.isnan(loss_diff)]
    if len(loss_diff) < 3 or np.isclose(loss_diff.std(ddof=1), 0):
        return {"dm_stat": float("nan"), "p_value": float("nan")}
    dm_stat = loss_diff.mean() / (loss_diff.std(ddof=1) / np.sqrt(len(loss_diff)))
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=len(loss_diff) - 1))
    return {"dm_stat": float(dm_stat), "p_value": float(p_value)}


def paired_tests(errors_by_model: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_a, model_b in combinations(errors_by_model, 2):
        a = np.asarray(errors_by_model[model_a], dtype=float)
        b = np.asarray(errors_by_model[model_b], dtype=float)
        n = min(len(a), len(b))
        if n < 3:
            continue
        a = a[:n]
        b = b[:n]
        dm = diebold_mariano_test(a, b)
        try:
            t_stat, t_p = stats.ttest_rel(a, b, nan_policy="omit")
        except Exception:
            t_stat, t_p = np.nan, np.nan
        try:
            w_stat, w_p = stats.wilcoxon(a, b)
        except Exception:
            w_stat, w_p = np.nan, np.nan
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "paired_t_stat": float(t_stat),
                "paired_t_p_value": float(t_p),
                "wilcoxon_stat": float(w_stat),
                "wilcoxon_p_value": float(w_p),
                **dm,
            }
        )
    return rows
