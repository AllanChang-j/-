from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backtesting.simulator import BacktestConfig, backtest_predictions
from evaluation.metrics import classification_signal, evaluate_predictions
from evaluation.reports import flatten_metrics, write_comparison_report, write_model_reports
from evaluation.statistics import paired_tests
from feature_engineering.build import build_feature_frame
from feature_selection.selectors import FeatureSelectionResult, select_features
from models.base import PredictionResult
from models.cnn.model import build_cnn
from models.lightgbm_model import LightGBMWrapper
from models.lstm.model import build_lstm
from models.transformer.model import build_transformer
from preprocessing.labels import add_prediction_labels, num_classes
from preprocessing.scaling import TabularPreprocessor
from preprocessing.windows import WindowDataset, make_sliding_windows, split_frame_by_dates
from training.torch_trainer import TorchTrainingConfig, predict_torch_model, train_torch_model
from utils.io import ensure_dir, save_json
from utils.reproducibility import get_device, set_global_seed
from validation.splitters import apply_date_split, rolling_window_splits, sklearn_time_series_splits, walk_forward_splits
from visualization.plots import (
    plot_confusion,
    plot_equity_curve,
    plot_feature_importance,
    plot_learning_curve,
    plot_prediction_vs_truth,
    plot_roc_pr,
    plot_rolling_risk,
)


def _task(config: dict[str, Any]) -> str:
    return str(config.get("labels", {}).get("task", "binary"))


def _torch_config(config: dict[str, Any]) -> TorchTrainingConfig:
    training = config.get("training", {})
    device_config = training.get("device", "auto")
    device = get_device(True) if device_config == "auto" else str(device_config)
    return TorchTrainingConfig(
        batch_size=int(training.get("batch_size", 64)),
        max_epochs=int(training.get("max_epochs", 30)),
        patience=int(training.get("patience", 5)),
        learning_rate=float(training.get("learning_rate", 0.001)),
        weight_decay=float(training.get("weight_decay", 0.0001)),
        gradient_clip=float(training.get("gradient_clip", 1.0)),
        scheduler=str(training.get("scheduler", "cosine")),
        device=device,
        num_workers=int(training.get("num_workers", 0)),
    )


def _backtest_config(config: dict[str, Any]) -> BacktestConfig:
    raw = config.get("backtest", {})
    return BacktestConfig(**{key: raw.get(key, getattr(BacktestConfig(), key)) for key in BacktestConfig.__dataclass_fields__})


def _model_size_bytes(obj: Any) -> int:
    with tempfile.NamedTemporaryFile() as tmp:
        joblib.dump(obj, tmp.name)
        return Path(tmp.name).stat().st_size


def _prediction_frame(result: PredictionResult, task: str, long_threshold: float) -> pd.DataFrame:
    frame = result.meta.copy()
    frame["target"] = result.targets
    frame["prediction"] = result.predictions
    if result.probabilities is not None and result.probabilities.ndim == 2:
        for index in range(result.probabilities.shape[1]):
            frame[f"prob_{index}"] = result.probabilities[:, index]
        score = result.probabilities[:, 1] if result.probabilities.shape[1] > 1 else result.probabilities[:, 0]
    else:
        score = result.predictions.astype(float)
    frame["score"] = score
    frame["signal"] = classification_signal(result.predictions, result.probabilities, task, threshold=long_threshold)
    return frame


def _classification_errors(result: PredictionResult, task: str) -> np.ndarray:
    if task == "regression":
        return (result.targets - result.predictions) ** 2
    if result.probabilities is not None and result.probabilities.shape[1] == 2:
        return -(result.targets * np.log(result.probabilities[:, 1] + 1e-12) + (1 - result.targets) * np.log(result.probabilities[:, 0] + 1e-12))
    return (result.targets != result.predictions).astype(float)


def prepare_research_frame(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    set_global_seed(int(config.get("experiment", {}).get("seed", 42)))
    features, candidate_columns = build_feature_frame(df, config)
    print(f"Built feature frame: rows={len(features)}, candidate_features={len(candidate_columns)}", flush=True)
    labels = config.get("labels", {})
    labeled = add_prediction_labels(
        features,
        horizon=int(labels.get("horizon", 5)),
        task=_task(config),
        price_column=str(config.get("data", {}).get("price_column", "adjusted_close")),
        neutral_threshold=float(labels.get("neutral_threshold", 0.01)),
    )
    print(f"Added labels: rows={len(labeled)}", flush=True)
    return labeled, candidate_columns


def prepare_window_data(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    candidate_columns: list[str],
    config: dict[str, Any],
    full_df: pd.DataFrame | None = None,
) -> tuple[WindowDataset, WindowDataset, WindowDataset, FeatureSelectionResult]:
    feature_cfg = config.get("features", {})
    configured_features = [str(column) for column in feature_cfg.get("selected_features", [])]
    if configured_features:
        missing = [column for column in configured_features if column not in candidate_columns]
        if missing:
            raise ValueError(f"Configured selected_features are missing from candidate features: {missing}")
        selected = configured_features[: int(feature_cfg.get("max_features", len(configured_features)))]
        selection = FeatureSelectionResult(
            selected_features=selected,
            ranking=pd.DataFrame({"feature": selected, "aggregate_rank": np.arange(1, len(selected) + 1)}),
            removed_low_quality=[],
            removed_correlated=[],
        )
    elif str(feature_cfg.get("selection_method", "ensemble")) == "none":
        selected = candidate_columns[: int(feature_cfg.get("max_features", len(candidate_columns)))]
        selection = FeatureSelectionResult(
            selected_features=selected,
            ranking=pd.DataFrame({"feature": selected, "aggregate_rank": np.arange(1, len(selected) + 1)}),
            removed_low_quality=[],
            removed_correlated=[],
        )
    else:
        selection = select_features(
            train_df=train_df,
            feature_columns=candidate_columns,
            task=_task(config),
            max_features=int(feature_cfg.get("max_features", 120)),
            min_non_null_ratio=float(feature_cfg.get("min_non_null_ratio", 0.85)),
            correlation_threshold=float(feature_cfg.get("correlation_threshold", 0.96)),
            random_state=int(config.get("experiment", {}).get("seed", 42)),
        )
    print(f"Selected features: count={len(selection.selected_features)}", flush=True)

    def compact_frame(frame: pd.DataFrame) -> pd.DataFrame:
        meta_columns = [
            "date",
            "symbol",
            "name",
            "market",
            "close",
            price_column,
            "target",
            "future_return",
            "execution_return",
            "signal_date",
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
        ]
        keep_columns = [column for column in dict.fromkeys([*meta_columns, *selection.selected_features]) if column in frame.columns]
        return frame.loc[:, keep_columns].copy()

    preprocessor = TabularPreprocessor(scale=bool(config.get("dataset", {}).get("scale_features", True)))
    sequence_length = int(config.get("dataset", {}).get("sequence_length", 30))
    price_column = str(config.get("data", {}).get("price_column", "adjusted_close"))
    compact_train = compact_frame(train_df)
    compact_full = compact_frame(full_df if full_df is not None else pd.concat([train_df, validation_df, test_df], axis=0, copy=False))
    preprocessor.fit(compact_train, selection.selected_features)
    print("Fitted fold-safe preprocessor on training split", flush=True)
    train_processed = preprocessor.transform(compact_train)
    full_processed = preprocessor.transform(compact_full)
    print(f"Transformed compact frames: train_rows={len(train_processed)}, full_rows={len(full_processed)}", flush=True)
    train_dates = set(pd.to_datetime(train_df["date"]).unique())
    validation_dates = set(pd.to_datetime(validation_df["date"]).unique())
    test_dates = set(pd.to_datetime(test_df["date"]).unique())
    print("Building train windows", flush=True)
    train_windows = make_sliding_windows(train_processed, selection.selected_features, sequence_length, price_column=price_column, target_dates=train_dates)
    print(f"Built train windows: samples={len(train_windows.y)}", flush=True)
    print("Building validation windows with historical context", flush=True)
    validation_windows = make_sliding_windows(full_processed, selection.selected_features, sequence_length, price_column=price_column, target_dates=validation_dates)
    print(f"Built validation windows: samples={len(validation_windows.y)}", flush=True)
    print("Building test windows with historical context", flush=True)
    test_windows = make_sliding_windows(full_processed, selection.selected_features, sequence_length, price_column=price_column, target_dates=test_dates)
    print(f"Built test windows: samples={len(test_windows.y)}", flush=True)
    return train_windows, validation_windows, test_windows, selection


def run_torch_experiment(
    model_name: str,
    train_data: WindowDataset,
    validation_data: WindowDataset,
    test_data: WindowDataset,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[PredictionResult, PredictionResult]:
    task = _task(config)
    n_outputs = num_classes(task)
    n_features = train_data.X.shape[2]
    model_config = config.get("models", {}).get(model_name, {})
    if model_name == "cnn":
        model = build_cnn(model_config, n_features=n_features, n_outputs=n_outputs, task=task)
    elif model_name == "lstm":
        model = build_lstm(model_config, n_features=n_features, n_outputs=n_outputs, task=task)
    elif model_name == "transformer":
        model = build_transformer(model_config, n_features=n_features, n_outputs=n_outputs, task=task)
    else:
        raise ValueError(f"Unsupported torch model: {model_name}")

    torch_config = _torch_config(config)
    params = {**model_config, **config.get("training", {})}
    validation_result = train_torch_model(
        model_name=model_name,
        model=model,
        train_data=train_data,
        validation_data=validation_data,
        task=task,
        config=torch_config,
        output_dir=output_dir,
        params=params,
    )
    validation_result.metrics = evaluate_predictions(
        validation_result.targets,
        validation_result.predictions,
        validation_result.probabilities,
        task,
    )
    test_result = predict_torch_model(
        model_name=model_name,
        model=model,
        dataset=test_data,
        task=task,
        config=torch_config,
        params=params,
    )
    test_result.training_time_sec = validation_result.training_time_sec
    test_result.history = validation_result.history
    test_result.metrics = evaluate_predictions(test_result.targets, test_result.predictions, test_result.probabilities, task)
    return validation_result, test_result


def run_lightgbm_experiment(
    train_data: WindowDataset,
    validation_data: WindowDataset,
    test_data: WindowDataset,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[PredictionResult, PredictionResult]:
    task = _task(config)
    model_config = dict(config.get("models", {}).get("lightgbm", {}))
    model_config.setdefault("random_state", int(config.get("experiment", {}).get("seed", 42)))
    model_config.pop("enabled", None)
    max_train_windows = model_config.pop("max_train_windows", None)
    wrapper = LightGBMWrapper(task=task, params=model_config)
    if max_train_windows is not None and int(max_train_windows) < len(train_data.y):
        rng = np.random.default_rng(int(model_config.get("random_state", 42)))
        train_indices = np.sort(rng.choice(len(train_data.y), size=int(max_train_windows), replace=False))
        X_train = train_data.X[train_indices].reshape(len(train_indices), train_data.X.shape[1] * train_data.X.shape[2])
        y_source = train_data.y[train_indices]
        print(f"Training tree baseline on sampled windows: samples={len(train_indices)}", flush=True)
    else:
        X_train = train_data.flatten()
        y_source = train_data.y
    y_train = y_source if task == "regression" else y_source.astype(int)
    X_valid = validation_data.flatten()
    y_valid = validation_data.y if task == "regression" else validation_data.y.astype(int)

    start = time.perf_counter()
    wrapper.fit(X_train, y_train, X_valid=X_valid, y_valid=y_valid)
    training_time = time.perf_counter() - start
    model_path = output_dir / "models" / "lightgbm.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper.save(str(model_path))
    size = _model_size_bytes(wrapper)

    def predict_dataset(dataset: WindowDataset) -> PredictionResult:
        X = dataset.flatten()
        start_infer = time.perf_counter()
        predictions = wrapper.predict(X)
        probabilities = wrapper.predict_proba(X)
        inference_time = time.perf_counter() - start_infer
        result = PredictionResult(
            model_name=wrapper.backend,
            predictions=np.asarray(predictions),
            probabilities=probabilities,
            targets=dataset.y,
            meta=dataset.meta.copy(),
            training_time_sec=training_time,
            inference_time_sec=inference_time,
            model_size_bytes=size,
            params={**model_config, "backend": wrapper.backend, "flatten_feature_names": train_data.flatten_feature_names()},
        )
        result.metrics = evaluate_predictions(result.targets, result.predictions, result.probabilities, task)
        return result

    return predict_dataset(validation_data), predict_dataset(test_data)


def _validation_splits(frame: pd.DataFrame, config: dict[str, Any]):
    validation_cfg = config.get("validation", {})
    method = str(validation_cfg.get("method", "walk_forward"))
    dates = pd.to_datetime(frame["date"]).unique()
    default_horizon = int(config.get("labels", {}).get("horizon", 1))
    purge = int(validation_cfg.get("purge", default_horizon))
    embargo = int(validation_cfg.get("embargo", default_horizon))
    if method == "rolling_window":
        return rolling_window_splits(
            dates,
            n_splits=int(validation_cfg.get("n_splits", 3)),
            train_window=int(validation_cfg.get("train_window", 252)),
            validation_window=int(validation_cfg.get("validation_window", 63)),
            step_size=int(validation_cfg.get("step_size", 63)),
            purge=purge,
            embargo=embargo,
        )
    if method == "time_series_split":
        return sklearn_time_series_splits(dates, n_splits=int(validation_cfg.get("n_splits", 3)), purge=purge)
    return walk_forward_splits(
        dates,
        n_splits=int(validation_cfg.get("n_splits", 3)),
        train_window=int(validation_cfg.get("train_window", 252)),
        validation_window=int(validation_cfg.get("validation_window", 63)),
        step_size=int(validation_cfg.get("step_size", 63)),
        expanding=bool(validation_cfg.get("expanding", True)),
        purge=purge,
        embargo=embargo,
    )


def run_cross_validation(
    cv_frame: pd.DataFrame,
    candidate_columns: list[str],
    config: dict[str, Any],
    output_dir: Path,
    enabled_models: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in _validation_splits(cv_frame, config):
        fold_train, fold_validation = apply_date_split(cv_frame, split)
        if fold_train.empty or fold_validation.empty:
            continue
        try:
            train_data, validation_data, test_data, selection = prepare_window_data(
                fold_train,
                fold_validation,
                fold_validation,
                candidate_columns,
                config,
                full_df=cv_frame[pd.to_datetime(cv_frame["date"]) <= pd.to_datetime(split.validation_dates[-1])].copy(),
            )
        except ValueError:
            continue
        for model_name in enabled_models:
            fold_dir = output_dir / "cross_validation" / split.name / model_name
            try:
                if model_name in {"cnn", "lstm", "transformer"}:
                    validation_result, _test_result = run_torch_experiment(
                        model_name,
                        train_data,
                        validation_data,
                        test_data,
                        config,
                        fold_dir,
                    )
                elif model_name == "lightgbm":
                    validation_result, _test_result = run_lightgbm_experiment(
                        train_data,
                        validation_data,
                        test_data,
                        config,
                        fold_dir,
                    )
                else:
                    continue
                row = flatten_metrics(validation_result.model_name, validation_result.metrics)
                row["fold"] = split.name
                row["train_start"] = str(split.train_dates[0])
                row["train_end"] = str(split.train_dates[-1])
                row["validation_start"] = str(split.validation_dates[0])
                row["validation_end"] = str(split.validation_dates[-1])
                row["selected_feature_count"] = len(selection.selected_features)
                row["training_time_sec"] = validation_result.training_time_sec
                rows.append(row)
            except Exception as exc:
                rows.append({"fold": split.name, "model": model_name, "error": str(exc)})
    cv_results = pd.DataFrame(rows)
    cv_output = output_dir / "reports" / "cross_validation_report.csv"
    cv_output.parent.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(cv_output, index=False)
    if not cv_results.empty:
        numeric = cv_results.select_dtypes(include=[np.number]).columns.tolist()
        if numeric:
            summary = cv_results.groupby("model")[numeric].agg(["mean", "std", "median", "min", "max"])
            summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
            summary.reset_index().to_csv(output_dir / "reports" / "cross_validation_summary.csv", index=False)
    return cv_results


def run_full_experiment(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    output_dir = ensure_dir(config.get("experiment", {}).get("output_dir", "experiments/default"))
    set_global_seed(int(config.get("experiment", {}).get("seed", 42)))
    research_frame, candidate_columns = prepare_research_frame(df, config)
    purge = int(config.get("validation", {}).get("purge", config.get("labels", {}).get("horizon", 1)))
    train_df, validation_df, test_df = split_frame_by_dates(
        research_frame,
        validation_size=float(config.get("dataset", {}).get("validation_size", 0.2)),
        test_size=float(config.get("dataset", {}).get("test_size", 0.2)),
        purge=purge,
    )
    train_data, validation_data, test_data, selection = prepare_window_data(train_df, validation_df, test_df, candidate_columns, config, full_df=research_frame)

    save_json(
        {
            "candidate_feature_count": len(candidate_columns),
            "selected_feature_count": len(selection.selected_features),
            "selected_features": selection.selected_features,
            "removed_low_quality": selection.removed_low_quality,
            "removed_correlated": selection.removed_correlated,
        },
        output_dir / "feature_selection_summary.json",
    )
    selection.ranking.to_csv(output_dir / "feature_selection_ranking.csv", index=False)
    plot_feature_importance(selection.ranking, output_dir / "figures" / "feature_importance.png")

    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    backtest_rows: list[dict[str, Any]] = []
    speed_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    errors_by_model: dict[str, np.ndarray] = {}
    task = _task(config)
    backtest_config = _backtest_config(config)

    enabled_models = [name for name, values in config.get("models", {}).items() if bool(values.get("enabled", True))]
    validation_config = config.get("validation", {})
    if bool(validation_config.get("run_cross_validation", True)):
        cv_frame = pd.concat([train_df, validation_df], axis=0, copy=False)
        print(f"Running independent date-based CV: rows={len(cv_frame)}, models={enabled_models}", flush=True)
        run_cross_validation(cv_frame, candidate_columns, config, Path(output_dir), enabled_models)
    else:
        print("Skipping cross validation for this run; holdout validation/test will still run.", flush=True)

    for model_name in enabled_models:
        print(f"Training final model: {model_name}", flush=True)
        if model_name in {"cnn", "lstm", "transformer"}:
            validation_result, test_result = run_torch_experiment(model_name, train_data, validation_data, test_data, config, Path(output_dir))
        elif model_name == "lightgbm":
            validation_result, test_result = run_lightgbm_experiment(train_data, validation_data, test_data, config, Path(output_dir))
        else:
            continue
        print(f"Completed final model: {model_name}", flush=True)

        prediction_frame = _prediction_frame(test_result, task, long_threshold=backtest_config.long_threshold)
        equity, trades, backtest_metrics = backtest_predictions(prediction_frame, backtest_config)
        report_model_name = test_result.model_name
        if task != "regression":
            for threshold in [0.45, 0.50, 0.55, 0.60, 0.65]:
                threshold_frame = _prediction_frame(test_result, task, long_threshold=threshold)
                _equity_s, _trades_s, sensitivity_metrics = backtest_predictions(threshold_frame, backtest_config)
                sensitivity_rows.append({"model": report_model_name, "long_threshold": threshold, **sensitivity_metrics})
        write_model_reports(
            output_dir=Path(output_dir) / "reports",
            model_name=report_model_name,
            validation_metrics=validation_result.metrics,
            test_metrics=test_result.metrics,
            backtest_metrics=backtest_metrics,
            predictions=prediction_frame,
            equity=equity,
            trades=trades,
            history=test_result.history,
            params=test_result.params,
        )

        figure_dir = Path(output_dir) / "figures" / report_model_name
        plot_learning_curve(test_result.history, figure_dir / "learning_curve.png", f"{report_model_name} learning curve")
        if task != "regression":
            plot_confusion(test_result.targets, test_result.predictions, figure_dir / "confusion_matrix.png", report_model_name)
            plot_roc_pr(test_result.targets, test_result.probabilities, figure_dir / "roc.png", figure_dir / "pr.png", report_model_name)
        plot_prediction_vs_truth(test_result.meta, test_result.targets, test_result.predictions, figure_dir / "prediction_vs_truth.png", report_model_name)
        plot_equity_curve(equity, figure_dir / "equity_curve.png", f"{report_model_name} equity curve")
        plot_rolling_risk(equity, figure_dir / "rolling_sharpe.png", figure_dir / "rolling_drawdown.png")

        validation_rows.append(flatten_metrics(report_model_name, validation_result.metrics))
        test_rows.append(flatten_metrics(report_model_name, test_result.metrics))
        backtest_rows.append(flatten_metrics(report_model_name, backtest_metrics))
        speed_rows.append(
            {
                "model": report_model_name,
                "training_time_sec": test_result.training_time_sec,
                "inference_time_sec": test_result.inference_time_sec,
                "model_size_bytes": test_result.model_size_bytes,
                "memory_usage_bytes_estimate": test_result.model_size_bytes,
                "params": test_result.params,
            }
        )
        errors_by_model[report_model_name] = _classification_errors(test_result, task)

    statistical_tests = paired_tests(errors_by_model, horizon=int(config.get("labels", {}).get("horizon", 1)))
    summary = write_comparison_report(
        output_dir=Path(output_dir) / "reports",
        validation_rows=validation_rows,
        test_rows=test_rows,
        backtest_rows=backtest_rows,
        speed_rows=speed_rows,
        statistical_tests=statistical_tests,
        sensitivity_rows=sensitivity_rows,
    )
    save_json(config, Path(output_dir) / "experiment_config.json")
    return summary
