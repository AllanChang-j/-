# Lab LightGBM Top Feature Experiments

This guide prepares the high-compute LightGBM comparison before running the deeper CNN/LSTM/Transformer plan.

## Goal

Run the same leakage-safe Taiwan stock dataset and preprocessing pipeline with three feature-selection sizes:

- Top 40 features
- Top 80 features
- Top 120 features

Only the LightGBM baseline is enabled. CNN, LSTM, and Transformer are disabled in these configs.

## Download On The Lab Machine

```bash
git clone https://github.com/AllanChang-j/-.git stock-dashboard
cd stock-dashboard
```

If the repository already exists:

```bash
cd stock-dashboard
git pull origin master
```

## Prepare Python Environment

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-research.txt
```

Confirm that the real LightGBM package is installed:

```bash
python - <<'PY'
import lightgbm
print(lightgbm.__version__)
PY
```

If this fails, install LightGBM before running the lab configs. Otherwise the framework will fall back to sklearn `HistGradientBoosting`, which is not the formal LightGBM baseline.

## Prepare Data

The configs expect:

```text
data/raw/taiwan_daily_ohlcv_20240101_20260630.csv
```

That raw data file is intentionally not committed to GitHub. Copy it from the MacBook or regenerate it with the project data collector before running experiments.

## Run One Experiment

Top 40:

```bash
python main.py --config config/lab_lightgbm_top40.yaml
```

Top 80:

```bash
python main.py --config config/lab_lightgbm_top80.yaml
```

Top 120:

```bash
python main.py --config config/lab_lightgbm_top120.yaml
```

## Run All Three

```bash
bash scripts/run_lab_lightgbm_top_features.sh
```

After all three runs complete, build one comparison CSV:

```bash
python scripts/summarize_lab_lightgbm_top_features.py
```

The summary is written to:

```text
experiments/lab_lightgbm_top_feature_summary.csv
```

## Outputs

Each config writes to a separate experiment folder:

```text
experiments/lab_lightgbm_top40_h5_seq30/
experiments/lab_lightgbm_top80_h5_seq30/
experiments/lab_lightgbm_top120_h5_seq30/
```

Main files to inspect:

```text
reports/final_comparison_report.xlsx
reports/test_comparison.csv
reports/backtest_comparison.csv
reports/cross_validation_report.csv
reports/cross_validation_summary.csv
feature_selection_summary.json
feature_selection_ranking.csv
```

## Expected Resource Use

The Top 120 experiment creates 30-day windows with 120 selected features, so each sample has 3,600 flattened LightGBM inputs. Use a machine with enough RAM. For full train plus validation/test evaluation, 32 GB may be tight; 64 GB is more comfortable.

The RTX 5050 is useful later for PyTorch CNN/LSTM/Transformer experiments. These LightGBM configs primarily use the i7-13700 CPU unless you intentionally install and configure a GPU-enabled LightGBM build.

## Safety Notes

- Cross validation is date-based and does not split different stocks from the same date across train and validation.
- Feature selection and preprocessing are refit inside each CV fold.
- Purge and embargo are both set to 5 trading days for the t+5 label.
- Final test data is not used for feature selection, scaling, threshold tuning, or model selection.
