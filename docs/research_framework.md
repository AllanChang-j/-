# Stock Prediction Research Framework

This framework compares four independent models for Taiwan daily stock trend prediction:

- CNN
- LSTM
- Transformer
- LightGBM baseline

The deep models and LightGBM use the same cleaned daily dataset, feature engineering pipeline, feature selection process, labels, and sliding-window samples. The models are trained independently so validation results are not mixed.

## Data Schema

Prepare a CSV with at least these columns:

```text
date,symbol,name,market,open,high,low,close,adjusted_close,volume,amount
```

Required columns are:

```text
date,symbol,open,high,low,close,volume
```

Optional columns:

- `adjusted_close`: defaults to `close` if missing
- `amount`
- fundamental columns prefixed with `fund_`, for example `fund_pe`, `fund_pb`, `fund_roe`
- macro columns prefixed with `macro_`, for example `macro_taiex_return`

Chinese column aliases such as `日期`, `代號`, `名稱`, `開盤價`, `最高價`, `最低價`, `收盤價`, `成交股數`, and `成交金額` are normalized automatically.

## Quick Start

Install research dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-research.txt
```

Run a fast smoke test. If the configured sample CSV does not exist, synthetic Taiwan-like OHLCV data will be generated under `data/sample/`.

```bash
python main.py --config config/research_default.yaml --fast --make-sample-data
```

Run on your real Taiwan daily data:

```bash
python main.py \
  --config config/research_default.yaml \
  --data data/raw/taiwan_daily_ohlcv.csv \
  --output-dir experiments/taiwan_daily_v1
```

Change task or horizon:

```bash
python main.py --data data/raw/taiwan_daily_ohlcv.csv --task three_class --horizon 10
python main.py --data data/raw/taiwan_daily_ohlcv.csv --task regression --horizon 20
```

## Leakage Prevention

The pipeline follows these rules:

- Rows are sorted by `symbol,date`.
- Technical indicators use rolling, expanding, or lagged values only.
- Labels use `future_return = adjusted_close[t+h] / adjusted_close[t] - 1`.
- Sliding-window sample `X[t-window+1:t]` predicts the target generated at `t`.
- Train/validation/test splits are by date and never shuffled.
- Feature selection is fitted only on the training period.
- Imputation and scaling are fitted only on the training period.
- Cross-validation excludes the final unseen test period.

## Feature Engineering

The feature factory creates roughly 100 to 200 candidate features depending on configured windows.

Included groups:

- Trend: MA, EMA, WMA, slope, linear regression slope, polynomial trend, rolling regression R2, Hurst exponent
- Momentum: RSI, stochastic, ROC, momentum, MACD, PPO
- Volatility: ATR, true range, historical volatility, Parkinson volatility, Bollinger width, Keltner width
- Volume: OBV, CMF, VWAP, volume MA, volume ratio, amount MA, amount ratio
- Statistical: rolling mean, std, median, skewness, kurtosis, quantiles
- Return: daily, weekly, monthly, log, cumulative, intraday, overnight
- Lag: return, volume ratio, close lags for 1, 2, 3, 5, 10, 20 days
- Calendar: weekday, month, quarter, month start/end, holiday flag

## Feature Selection

Feature selection is automatic and training-period only:

- Remove low-quality features with too many missing values
- Remove zero-variance features
- Remove highly correlated variables
- Rank remaining candidates with Lasso/L1, Elastic Net, Mutual Information, tree importance, Permutation Importance
- Use SHAP importance when `shap` is installed
- Select top `features.max_features`

Outputs:

```text
feature_selection_summary.json
feature_selection_ranking.csv
figures/feature_importance.png
```

## Models

### CNN

Default is 1D CNN over time:

- Conv1D
- ReLU
- BatchNorm
- Pooling
- Dropout
- Dense output head

The code also includes a `candlestick_image` CNN variant that treats the window as a 2D time-feature image.

### LSTM

Many-to-one LSTM:

- single or multi layer
- optional bidirectional mode
- optional attention pooling
- dropout
- LayerNorm output head

### Transformer

Encoder-only time-series Transformer:

- input projection
- positional encoding
- multi-head attention
- feed-forward blocks
- residual connection and LayerNorm through PyTorch encoder layers

### LightGBM

LightGBM is the fourth model and the critical financial baseline. It uses the same sliding-window dataset flattened into tabular features. If `lightgbm` is not installed, the framework falls back to sklearn `HistGradientBoosting` and records that backend in `best_params.json`.

## Validation

Supported validation methods:

- Walk-forward validation
- Rolling-window validation
- Expanding-window validation
- sklearn `TimeSeriesSplit`

Each model runs validation independently. No random K-Fold is used.

CV outputs:

```text
reports/cross_validation_report.csv
reports/cross_validation_summary.csv
```

## Backtesting

The simulator supports:

- commission
- transaction tax
- slippage
- position sizing
- cash management
- stop loss
- take profit
- maximum positions
- long-only or optional short mode

Outputs:

```text
reports/<model>/equity_curve.csv
reports/<model>/trades.csv
reports/<model>/backtest_metrics.json
figures/<model>/equity_curve.png
figures/<model>/rolling_sharpe.png
figures/<model>/rolling_drawdown.png
```

## Reports

Each run writes:

- training artifacts
- validation report
- test report
- cross-validation report
- backtesting report
- final comparison report
- best model summary
- publication-style figures
- reproducible config logs
- model files

Main output paths:

```text
experiments/<run_name>/experiment_config.json
experiments/<run_name>/reports/final_comparison_report.xlsx
experiments/<run_name>/reports/final_comparison_report.md
experiments/<run_name>/reports/best_model_summary.json
experiments/<run_name>/reports/validation_comparison.csv
experiments/<run_name>/reports/test_comparison.csv
experiments/<run_name>/reports/backtest_comparison.csv
experiments/<run_name>/reports/speed_model_size_comparison.csv
experiments/<run_name>/reports/robustness_generalization_overfitting.csv
experiments/<run_name>/reports/sensitivity_analysis.csv
experiments/<run_name>/reports/statistical_significance_tests.csv
```

## Statistical Tests

The comparison report includes pairwise tests where sample sizes allow:

- Diebold-Mariano test
- paired t-test
- Wilcoxon signed-rank test

For classification, the loss proxy is log loss when probabilities are available; otherwise classification error is used. For regression, squared forecast error is used.

## Robustness and Sensitivity

The framework exports:

- cross-validation mean and standard deviation by model
- validation-to-test generalization gaps
- overfitting risk labels based on F1 gap
- threshold sensitivity analysis for classification trading signals

These tables are intended to answer the practical research question: whether CNN, LSTM, or Transformer complexity adds value beyond the LightGBM baseline.
