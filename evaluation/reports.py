from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from utils.io import save_json


def flatten_metrics(model_name: str, metrics: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    row: dict[str, Any] = {"model": model_name}
    for key, value in metrics.items():
        if isinstance(value, (dict, list)):
            continue
        row[f"{prefix}{key}"] = value
    return row


def write_model_reports(
    output_dir: str | Path,
    model_name: str,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    backtest_metrics: dict[str, Any],
    predictions: pd.DataFrame,
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    history: dict[str, list[float]],
    params: dict[str, Any],
) -> None:
    model_dir = Path(output_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    save_json(validation_metrics, model_dir / "validation_metrics.json")
    save_json(test_metrics, model_dir / "test_metrics.json")
    save_json(backtest_metrics, model_dir / "backtest_metrics.json")
    save_json(params, model_dir / "best_params.json")
    predictions.to_csv(model_dir / "predictions.csv", index=False)
    trend_indicator = build_trend_indicator_table(predictions)
    if not trend_indicator.empty:
        trend_indicator.to_csv(model_dir / "trend_indicator.csv", index=False)
    equity.to_csv(model_dir / "equity_curve.csv", index=False)
    trades.to_csv(model_dir / "trades.csv", index=False)
    if history:
        pd.DataFrame(history).to_csv(model_dir / "learning_curve.csv", index=False)


def build_trend_indicator_table(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    frame = predictions.copy()
    output = pd.DataFrame()
    for column in ["date", "symbol", "name", "market", "close", "future_return", "execution_return"]:
        if column in frame.columns:
            output[column] = frame[column]
    if "prob_1" in frame.columns:
        prob_up = pd.to_numeric(frame["prob_1"], errors="coerce")
        output["trend_score"] = prob_up
        output["prob_up"] = prob_up
        output["trend_strength"] = (prob_up - 0.5).abs() * 2
        output["trend_label"] = pd.cut(
            prob_up,
            bins=[-float("inf"), 0.45, 0.55, float("inf")],
            labels=["down", "neutral", "up"],
        ).astype(str)
    elif "prediction" in frame.columns:
        prediction = pd.to_numeric(frame["prediction"], errors="coerce")
        output["trend_score"] = prediction
        output["predicted_future_return"] = prediction
        output["trend_strength"] = prediction.abs()
        output["trend_label"] = pd.cut(
            prediction,
            bins=[-float("inf"), -0.005, 0.005, float("inf")],
            labels=["down", "neutral", "up"],
        ).astype(str)
    if "target" in frame.columns:
        output["target"] = frame["target"]
    if "date" in output.columns and "trend_score" in output.columns:
        grouped_score = output.groupby("date")["trend_score"]
        output["daily_trend_percentile"] = grouped_score.rank(method="average", pct=True)
        output["daily_rank"] = grouped_score.rank(method="first", ascending=False).astype(int)
        output["daily_trend_label"] = pd.cut(
            output["daily_trend_percentile"],
            bins=[-float("inf"), 0.2, 0.8, float("inf")],
            labels=["down", "neutral", "up"],
        ).astype(str)
    return output


def write_comparison_report(
    output_dir: str | Path,
    validation_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    backtest_rows: list[dict[str, Any]],
    speed_rows: list[dict[str, Any]],
    statistical_tests: list[dict[str, Any]],
    sensitivity_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    validation_df = pd.DataFrame(validation_rows)
    test_df = pd.DataFrame(test_rows)
    backtest_df = pd.DataFrame(backtest_rows)
    speed_df = pd.DataFrame(speed_rows)
    stats_df = pd.DataFrame(statistical_tests)
    sensitivity_df = pd.DataFrame(sensitivity_rows or [])
    robustness_df = build_robustness_generalization_table(validation_df, test_df)

    validation_df.to_csv(output / "validation_comparison.csv", index=False)
    test_df.to_csv(output / "test_comparison.csv", index=False)
    backtest_df.to_csv(output / "backtest_comparison.csv", index=False)
    speed_df.to_csv(output / "speed_model_size_comparison.csv", index=False)
    stats_df.to_csv(output / "statistical_significance_tests.csv", index=False)
    sensitivity_df.to_csv(output / "sensitivity_analysis.csv", index=False)
    robustness_df.to_csv(output / "robustness_generalization_overfitting.csv", index=False)

    with pd.ExcelWriter(output / "final_comparison_report.xlsx") as writer:
        validation_df.to_excel(writer, sheet_name="validation", index=False)
        test_df.to_excel(writer, sheet_name="test", index=False)
        backtest_df.to_excel(writer, sheet_name="backtest", index=False)
        speed_df.to_excel(writer, sheet_name="speed_size", index=False)
        stats_df.to_excel(writer, sheet_name="statistics", index=False)
        sensitivity_df.to_excel(writer, sheet_name="sensitivity", index=False)
        robustness_df.to_excel(writer, sheet_name="robustness", index=False)

    best_model = None
    if not test_df.empty:
        score_column = "f1" if "f1" in test_df.columns else "rmse"
        ascending = score_column == "rmse"
        best_model = test_df.sort_values(score_column, ascending=ascending).iloc[0].to_dict()

    summary = {
        "best_model": best_model,
        "validation_models": validation_df.to_dict(orient="records"),
        "test_models": test_df.to_dict(orient="records"),
        "backtest_models": backtest_df.to_dict(orient="records"),
        "statistical_tests": stats_df.to_dict(orient="records"),
        "sensitivity_analysis": sensitivity_df.to_dict(orient="records"),
        "robustness_generalization": robustness_df.to_dict(orient="records"),
    }
    save_json(summary, output / "best_model_summary.json")
    write_markdown_summary(output / "final_comparison_report.md", summary)
    return summary


def build_robustness_generalization_table(validation_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    if validation_df.empty or test_df.empty or "model" not in validation_df or "model" not in test_df:
        return pd.DataFrame()
    merged = validation_df.merge(test_df, on="model", how="inner", suffixes=("_validation", "_test"))
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        output = {"model": row["model"]}
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "rmse", "mae"]:
            validation_key = f"{metric}_validation"
            test_key = f"{metric}_test"
            if validation_key in row and test_key in row:
                output[f"{metric}_generalization_gap"] = row[validation_key] - row[test_key]
                output[f"{metric}_absolute_gap"] = abs(row[validation_key] - row[test_key])
        if "f1_absolute_gap" in output:
            output["overfitting_risk"] = "high" if output["f1_absolute_gap"] > 0.10 else "moderate" if output["f1_absolute_gap"] > 0.05 else "low"
        rows.append(output)
    return pd.DataFrame(rows)


def write_markdown_summary(path: str | Path, summary: dict[str, Any]) -> None:
    lines = ["# Final Model Comparison Report", ""]
    best = summary.get("best_model")
    if best:
        lines.append("## Best Model")
        lines.append("")
        lines.append(f"- Model: `{best.get('model')}`")
        for key, value in best.items():
            if key != "model":
                lines.append(f"- {key}: {value}")
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- All models use the same leakage-safe preprocessing, labels, features, and sliding-window dataset.")
    lines.append("- CNN, LSTM, Transformer, and LightGBM are trained independently.")
    lines.append("- Time-series validation never shuffles observations.")
    lines.append("- Backtest metrics include costs, slippage, position sizing, max positions, stop loss, and take profit.")
    lines.append("- Robustness, generalization gap, overfitting risk, sensitivity, and statistical significance tables are exported.")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
