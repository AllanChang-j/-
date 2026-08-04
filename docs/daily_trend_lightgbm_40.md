# Daily Trend LightGBM 40

This is the daily-run candidate for a simple price-trend indicator.

## Purpose

Train one LightGBM model using about 40 fixed time-series technical features and output a table-friendly trend score.

Prediction target:

```text
adjusted_close[t+5] > adjusted_close[t]
```

Main table output:

```text
reports/lightgbm/trend_indicator.csv
```

Important columns:

- `trend_score`: probability of t+5 up direction
- `prob_up`: same as `trend_score`
- `trend_strength`: distance from 0.5, scaled to 0-1
- `trend_label`: `down`, `neutral`, or `up`
- `future_return`: realized t+5 return for historical evaluation
- `execution_return`: approximate t+1 open to t+6 open return for backtest alignment

## Why This Is Lighter

- No deep models
- No 20/30-day neural sequence tensor
- No feature-selection ensemble
- Sequence length is 1
- Lagged and rolling TA features carry the time-series information
- 3-fold date-based CV remains enabled

## Run

Windows:

```powershell
.\.venv\Scripts\python.exe main.py --config config/daily_trend_lightgbm_40.yaml
```

macOS / Linux:

```bash
python main.py --config config/daily_trend_lightgbm_40.yaml
```

## Outputs

```text
experiments/daily_trend_lightgbm_40_h5/reports/test_comparison.csv
experiments/daily_trend_lightgbm_40_h5/reports/backtest_comparison.csv
experiments/daily_trend_lightgbm_40_h5/reports/cross_validation_summary.csv
experiments/daily_trend_lightgbm_40_h5/reports/lightgbm/trend_indicator.csv
experiments/daily_trend_lightgbm_40_h5/reports/final_comparison_report.xlsx
```
