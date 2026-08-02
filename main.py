from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from data.loaders import load_daily_csv
from data.sample_generator import make_synthetic_taiwan_daily
from training.pipeline import run_full_experiment
from utils.config import deep_update, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Taiwan stock prediction research framework.")
    parser.add_argument("--config", default="config/research_default.yaml", help="YAML/JSON experiment config path.")
    parser.add_argument("--data", help="Override daily OHLCV CSV path.")
    parser.add_argument("--output-dir", help="Override experiment output directory.")
    parser.add_argument("--task", choices=["binary", "three_class", "regression"], help="Override prediction task.")
    parser.add_argument("--horizon", type=int, help="Override prediction horizon, e.g. 1, 5, 10, 20.")
    parser.add_argument("--sequence-length", type=int, help="Override sliding-window length.")
    parser.add_argument("--fast", action="store_true", help="Run a short smoke test with fewer epochs/features/folds.")
    parser.add_argument("--make-sample-data", action="store_true", help="Generate synthetic sample data then run.")
    return parser.parse_args()


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.data:
        overrides = deep_update(overrides, {"data": {"input_path": args.data, "make_sample_if_missing": False}})
    if args.output_dir:
        overrides = deep_update(overrides, {"experiment": {"output_dir": args.output_dir}})
    if args.task:
        overrides = deep_update(overrides, {"labels": {"task": args.task}})
    if args.horizon:
        overrides = deep_update(overrides, {"labels": {"horizon": args.horizon}})
    if args.sequence_length:
        overrides = deep_update(overrides, {"dataset": {"sequence_length": args.sequence_length}})
    if args.fast:
        overrides = deep_update(
            overrides,
            {
                "experiment": {"output_dir": args.output_dir or "experiments/fast_smoke"},
                "features": {"max_features": 40},
                "validation": {"n_splits": 1, "train_window": 180, "validation_window": 40, "step_size": 40},
                "training": {"max_epochs": 2, "patience": 1, "batch_size": 128, "device": "cpu"},
                "models": {
                    "cnn": {"channels": [16, 32]},
                    "lstm": {"hidden_size": 32, "attention": True},
                    "transformer": {"d_model": 32, "n_heads": 4, "num_layers": 1, "dim_feedforward": 64},
                    "lightgbm": {"n_estimators": 80},
                },
            },
        )
    return deep_update(config, overrides)


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    data_path = Path(config["data"]["input_path"])
    if args.make_sample_data or (config["data"].get("make_sample_if_missing", False) and not data_path.exists()):
        make_synthetic_taiwan_daily(data_path, seed=int(config.get("experiment", {}).get("seed", 42)))
    df = load_daily_csv(data_path)
    summary = run_full_experiment(df, config)
    print("Experiment complete.")
    print(f"Output: {config['experiment']['output_dir']}")
    if summary.get("best_model"):
        print(f"Best model: {summary['best_model'].get('model')}")


if __name__ == "__main__":
    main()
