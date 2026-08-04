# Level 1 Hardware Plan

This branch contains the higher-compute Level 1 experiment plan for a lab machine such as RTX 5050 plus i7-13700.

## Scope

Level 1 is the first serious high-compute run before expanding into nested CV, multi-horizon experiments, and larger Optuna searches.

It uses:

- Top 80 selected features
- t+5 binary up/down label
- 30-day input window
- 3-fold purged walk-forward validation
- purge = 5 trading days
- embargo = 5 trading days
- CNN, LSTM, Transformer, and LightGBM all enabled
- Deep models trained up to 30 epochs with early stopping
- Full LightGBM baseline with validation early stopping

## Branch

```bash
git clone https://github.com/AllanChang-j/-.git stock-dashboard
cd stock-dashboard
git switch codex/level1-hardware-plan
```

If the repository already exists:

```bash
cd stock-dashboard
git fetch origin
git switch codex/level1-hardware-plan
git pull origin codex/level1-hardware-plan
```

## Environment

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-research.txt
```

For NVIDIA GPU training, install a CUDA-enabled PyTorch build appropriate for the lab machine before running the experiment. Then verify:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
```

Verify real LightGBM is installed:

```bash
python - <<'PY'
import lightgbm
print("lightgbm", lightgbm.__version__)
PY
```

If LightGBM import fails, do not treat the tree result as the formal LightGBM baseline. The framework will fall back to sklearn only as a compatibility path.

## Data

The config expects:

```text
data/raw/taiwan_daily_ohlcv_20240101_20260630.csv
```

This raw data file is not committed to GitHub. Copy it from the MacBook or regenerate it before running.

## Run

```bash
bash scripts/run_level1_hardware_full_comparison.sh
```

Equivalent direct command:

```bash
python main.py --config config/level1_hardware_full_comparison.yaml
```

## Output

Results are written to:

```text
experiments/level1_hardware_full_comparison_h5_seq30_top80/
```

Key files:

```text
reports/final_comparison_report.xlsx
reports/test_comparison.csv
reports/backtest_comparison.csv
reports/cross_validation_report.csv
reports/cross_validation_summary.csv
reports/speed_model_size_comparison.csv
reports/statistical_significance_tests.csv
feature_selection_summary.json
feature_selection_ranking.csv
figures/
models/
tensorboard/
```

## Expected Runtime

This config is intentionally heavier than the MacBook baseline:

- 3-fold CV is enabled.
- Each CV fold refits feature selection and preprocessing.
- Four models run independently.
- Deep models can train up to 30 epochs.
- LightGBM uses the full training windows with early stopping.

On RTX 5050 plus i7-13700, this should be treated as a multi-hour experiment, not an interactive smoke test.

## Success Criteria

Do not judge only by accuracy. Inspect:

- test ROC-AUC
- test PR-AUC
- MCC
- F1
- cross-validation mean/std
- backtest cumulative return
- Sharpe ratio
- maximum drawdown
- trade count
- generalization gap

If all models remain below roughly 0.52 ROC-AUC and backtests remain strongly negative, the next step should be feature/label/trading-rule redesign rather than simply making the models larger.
