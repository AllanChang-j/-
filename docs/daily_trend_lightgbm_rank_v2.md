# Daily Trend LightGBM Rank V2

This version is designed as a daily table indicator rather than a trade-trigger model.

## Target

The model predicts each stock's same-day cross-sectional percentile of future t+5 return:

```text
target = percentile_rank(future_return[t+5] within the same date)
```

So a prediction near 1.0 means the stock is expected to rank near the top of the daily universe, while a prediction near 0.0 means it is expected to rank near the bottom.

## Model

- LightGBM regression
- 80 fixed time-series technical features
- sequence length 1
- 3-fold purged date-based walk-forward CV
- final train / validation / test split

## Main Output

```text
experiments/daily_trend_lightgbm_rank_v2_h5/reports/lightgbm/trend_indicator.csv
```

Recommended table columns:

- `daily_rank`
- `daily_trend_percentile`
- `daily_trend_label`
- `trend_score`
- `trend_strength`

## Bucket Report

```text
experiments/daily_trend_lightgbm_rank_v2_h5/reports/lightgbm/trend_bucket_report.csv
```

Use this to judge whether the indicator separates stronger and weaker stocks:

- top 5%
- top 10%
- top 20%
- middle 60%
- bottom 20%
- bottom 10%
- bottom 5%
- daily rank IC

## Run

Windows:

```powershell
.\.venv\Scripts\python.exe main.py --config config/daily_trend_lightgbm_rank_v2.yaml
```

macOS / Linux:

```bash
python main.py --config config/daily_trend_lightgbm_rank_v2.yaml
```
