from __future__ import annotations

from pathlib import Path

import pandas as pd


EXPERIMENTS = {
    40: Path("experiments/lab_lightgbm_top40_h5_seq30"),
    80: Path("experiments/lab_lightgbm_top80_h5_seq30"),
    120: Path("experiments/lab_lightgbm_top120_h5_seq30"),
}


def read_first_row(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def main() -> None:
    rows: list[dict[str, object]] = []
    for top_n, experiment_dir in EXPERIMENTS.items():
        reports_dir = experiment_dir / "reports"
        test = read_first_row(reports_dir / "test_comparison.csv")
        backtest = read_first_row(reports_dir / "backtest_comparison.csv")
        speed = read_first_row(reports_dir / "speed_model_size_comparison.csv")
        cv_summary_path = reports_dir / "cross_validation_summary.csv"
        cv_summary = {}
        if cv_summary_path.exists():
            cv_frame = pd.read_csv(cv_summary_path)
            if not cv_frame.empty:
                cv_summary = cv_frame.iloc[0].to_dict()
        rows.append(
            {
                "top_features": top_n,
                "experiment_dir": str(experiment_dir),
                "model": test.get("model"),
                "test_accuracy": test.get("accuracy"),
                "test_balanced_accuracy": test.get("balanced_accuracy"),
                "test_precision": test.get("precision"),
                "test_recall": test.get("recall"),
                "test_f1": test.get("f1"),
                "test_roc_auc": test.get("roc_auc"),
                "test_pr_auc": test.get("pr_auc"),
                "test_mcc": test.get("mcc"),
                "test_brier_score": test.get("brier_score"),
                "cumulative_return": backtest.get("cumulative_return"),
                "annual_return": backtest.get("annual_return"),
                "sharpe_ratio": backtest.get("sharpe_ratio"),
                "maximum_drawdown": backtest.get("maximum_drawdown"),
                "trade_count": backtest.get("trade_count"),
                "training_time_sec": speed.get("training_time_sec"),
                "inference_time_sec": speed.get("inference_time_sec"),
                "cv_f1_mean": cv_summary.get("f1_mean"),
                "cv_f1_std": cv_summary.get("f1_std"),
                "cv_roc_auc_mean": cv_summary.get("roc_auc_mean"),
                "cv_roc_auc_std": cv_summary.get("roc_auc_std"),
                "cv_pr_auc_mean": cv_summary.get("pr_auc_mean"),
                "cv_pr_auc_std": cv_summary.get("pr_auc_std"),
            }
        )
    output = Path("experiments/lab_lightgbm_top_feature_summary.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
