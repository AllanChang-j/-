from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
from scipy import stats


def _newey_west_variance(values: np.ndarray, max_lag: int) -> float:
    centered = values - values.mean()
    n = len(centered)
    gamma0 = float(np.dot(centered, centered) / n)
    variance = gamma0
    for lag in range(1, min(max_lag, n - 1) + 1):
        weight = 1 - lag / (max_lag + 1)
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        variance += 2 * weight * gamma
    return max(variance, 1e-12)


def diebold_mariano_test(loss_a: np.ndarray, loss_b: np.ndarray, horizon: int = 1) -> dict[str, float]:
    loss_diff = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    loss_diff = loss_diff[~np.isnan(loss_diff)]
    if len(loss_diff) < 3 or np.isclose(loss_diff.std(ddof=1), 0):
        return {"dm_stat": float("nan"), "p_value": float("nan")}
    nw_variance = _newey_west_variance(loss_diff, max(0, horizon - 1))
    dm_stat = loss_diff.mean() / np.sqrt(nw_variance / len(loss_diff))
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=len(loss_diff) - 1))
    return {"dm_newey_west_stat": float(dm_stat), "dm_newey_west_p_value": float(p_value)}


def block_bootstrap_mean_diff(loss_a: np.ndarray, loss_b: np.ndarray, block_size: int, n_bootstrap: int = 1000, seed: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    diff = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    if n < max(3, block_size):
        return {"block_bootstrap_mean_diff": float("nan"), "block_bootstrap_p_value": float("nan")}
    observed = float(diff.mean())
    block_starts = np.arange(0, n - block_size + 1)
    samples = []
    for _ in range(n_bootstrap):
        pieces = []
        while sum(len(piece) for piece in pieces) < n:
            start = int(rng.choice(block_starts))
            pieces.append(diff[start : start + block_size])
        sample = np.concatenate(pieces)[:n]
        samples.append(sample.mean())
    samples_array = np.asarray(samples)
    p_value = float(np.mean(np.abs(samples_array - samples_array.mean()) >= abs(observed)))
    return {"block_bootstrap_mean_diff": observed, "block_bootstrap_p_value": p_value}


def paired_tests(errors_by_model: dict[str, np.ndarray], horizon: int = 1) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_a, model_b in combinations(errors_by_model, 2):
        a = np.asarray(errors_by_model[model_a], dtype=float)
        b = np.asarray(errors_by_model[model_b], dtype=float)
        n = min(len(a), len(b))
        if n < 3:
            continue
        a = a[:n]
        b = b[:n]
        dm = diebold_mariano_test(a, b, horizon=horizon)
        bootstrap = block_bootstrap_mean_diff(a, b, block_size=max(1, horizon))
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
                **bootstrap,
            }
        )
    return rows
