# Lab One-Click Level 1 Run

This is the one-command path for a fresh lab machine.

It prepares the environment, installs research dependencies, verifies GPU and LightGBM, prepares data if needed, runs safety tests, trains Level 1 models, and writes final validation/test/backtest reports.

## One Command From A Fresh Clone

```bash
git clone https://github.com/AllanChang-j/-.git stock-dashboard && cd stock-dashboard && bash scripts/lab_one_click_level1.sh
```

If the repository already exists:

```bash
cd stock-dashboard && git pull origin master && bash scripts/lab_one_click_level1.sh
```

## What The Script Does

The script runs:

1. Create `.venv`
2. Upgrade `pip`, `setuptools`, and `wheel`
3. Install CUDA PyTorch when `nvidia-smi` is available
4. Install `requirements-research.txt`
5. Print environment diagnostics
6. Check `data/raw/taiwan_daily_ohlcv_20240101_20260630.csv`
7. If missing, collect official history from `2024-01-01` to `2026-06-30`
8. Validate the daily data file
9. Run `pytest tests`
10. Run Level 1 training and final test:

```bash
python main.py --config config/level1_hardware_full_comparison.yaml
```

## Default Level 1 Config

```text
config/level1_hardware_full_comparison.yaml
```

This config uses:

- Top 80 features
- t+5 binary label
- 30-day input window
- 3-fold purged walk-forward validation
- CNN, LSTM, Transformer, and LightGBM
- Deep models up to 30 epochs
- LightGBM full baseline with early stopping

## Data

Default expected file:

```text
data/raw/taiwan_daily_ohlcv_20240101_20260630.csv
```

If you already copied the CSV from the MacBook, the script skips collection.

If the file does not exist, the script attempts to collect it:

```bash
python data/collect_stage1_history.py \
  --start 2024-01-01 \
  --end 2026-06-30 \
  --output data/raw/taiwan_daily_ohlcv_20240101_20260630.csv \
  --strict-network
```

## Useful Overrides

Use an existing data file:

```bash
DATA_PATH=/path/to/taiwan_daily.csv bash scripts/lab_one_click_level1.sh
```

Skip data collection and fail if the CSV is missing:

```bash
COLLECT_IF_MISSING=0 bash scripts/lab_one_click_level1.sh
```

Run a different config:

```bash
CONFIG_PATH=config/lab_lightgbm_top80.yaml bash scripts/lab_one_click_level1.sh
```

Skip unit tests:

```bash
RUN_TESTS=0 bash scripts/lab_one_click_level1.sh
```

Force CUDA PyTorch install:

```bash
INSTALL_CUDA_TORCH=1 bash scripts/lab_one_click_level1.sh
```

Use a different CUDA PyTorch wheel index:

```bash
CUDA_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 bash scripts/lab_one_click_level1.sh
```

## Outputs

Logs:

```text
logs/level1_one_click_YYYYMMDD_HHMMSS.log
```

Experiment output:

```text
experiments/level1_hardware_full_comparison_h5_seq30_top80/
```

Key reports:

```text
reports/final_comparison_report.xlsx
reports/test_comparison.csv
reports/backtest_comparison.csv
reports/cross_validation_summary.csv
reports/best_model_summary.json
```

## Notes

- The script does not install NVIDIA drivers. The lab computer must already have a working driver for CUDA GPU training.
- If `lightgbm` cannot import, do not treat tree results as the formal LightGBM baseline.
- This is a long experiment. On RTX 5050 plus i7-13700, expect a multi-hour run.
