from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def _prepare(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_learning_curve(history: dict[str, list[float]], path: str | Path, title: str) -> None:
    if not history:
        return
    output_path = _prepare(path)
    plt.figure(figsize=(7, 4))
    for key, values in history.items():
        plt.plot(values, label=key)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, path: str | Path, title: str) -> None:
    output_path = _prepare(path)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def plot_roc_pr(y_true: np.ndarray, probabilities: np.ndarray | None, roc_path: str | Path, pr_path: str | Path, title: str) -> None:
    if probabilities is None or probabilities.ndim != 2 or probabilities.shape[1] != 2:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, probabilities[:, 1], ax=ax)
    ax.set_title(f"ROC - {title}")
    plt.tight_layout()
    plt.savefig(_prepare(roc_path))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, probabilities[:, 1], ax=ax)
    ax.set_title(f"PR - {title}")
    plt.tight_layout()
    plt.savefig(_prepare(pr_path))
    plt.close(fig)


def plot_feature_importance(ranking: pd.DataFrame, path: str | Path, top_n: int = 30) -> None:
    if ranking.empty or "aggregate_rank" not in ranking:
        return
    output_path = _prepare(path)
    frame = ranking.sort_values("aggregate_rank").head(top_n).sort_values("aggregate_rank", ascending=False)
    plt.figure(figsize=(8, max(5, top_n * 0.25)))
    sns.barplot(data=frame, x="aggregate_rank", y="feature", color="#3E7CB1")
    plt.title("Feature Importance Ranking")
    plt.xlabel("Aggregate rank (lower is better)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_prediction_vs_truth(meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, path: str | Path, title: str) -> None:
    output_path = _prepare(path)
    frame = meta.copy()
    frame["target"] = y_true
    frame["prediction"] = y_pred
    daily = frame.groupby("date")[["target", "prediction"]].mean().reset_index()
    plt.figure(figsize=(9, 4))
    plt.plot(daily["date"], daily["target"], label="Ground truth", linewidth=1.5)
    plt.plot(daily["date"], daily["prediction"], label="Prediction", linewidth=1.5)
    plt.title(title)
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_equity_curve(equity: pd.DataFrame, path: str | Path, title: str) -> None:
    if equity.empty:
        return
    output_path = _prepare(path)
    plt.figure(figsize=(9, 4))
    plt.plot(pd.to_datetime(equity["date"]), equity["equity"], linewidth=1.6)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_rolling_risk(equity: pd.DataFrame, sharpe_path: str | Path, drawdown_path: str | Path, window: int = 63) -> None:
    if equity.empty or len(equity) < 3:
        return
    frame = equity.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    returns = frame["daily_return"].fillna(0)
    rolling_sharpe = returns.rolling(window).mean() / (returns.rolling(window).std() + 1e-12) * np.sqrt(252)
    running_max = frame["equity"].cummax()
    drawdown = frame["equity"] / running_max - 1

    plt.figure(figsize=(9, 4))
    plt.plot(frame["date"], rolling_sharpe)
    plt.title("Rolling Sharpe")
    plt.tight_layout()
    plt.savefig(_prepare(sharpe_path))
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(frame["date"], drawdown)
    plt.title("Rolling Drawdown")
    plt.tight_layout()
    plt.savefig(_prepare(drawdown_path))
    plt.close()


def plot_attention_map(attention: np.ndarray | None, path: str | Path, title: str) -> None:
    if attention is None:
        return
    output_path = _prepare(path)
    plt.figure(figsize=(7, 3))
    sns.heatmap(np.atleast_2d(attention), cmap="viridis", cbar=True)
    plt.title(title)
    plt.xlabel("Time step")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
