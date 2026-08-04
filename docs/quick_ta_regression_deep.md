# Quick TA Regression Deep Test

This is the fast test requested after the Level 1 run proved too slow.

## Goal

Use ordinary CNN, LSTM, and Transformer models with a small set of common technical-analysis indicators as inputs, then predict a numeric future return.

Target:

```text
future_return = adjusted_close[t+5] / adjusted_close[t] - 1
```

The model output is therefore a predicted up/down magnitude, not a class label.

## What Is Different From Level 1

- Uses `features.profile: common_ta`
- Uses only common indicators such as returns, MA/EMA distance, RSI, MACD, stochastic, ATR, Bollinger width, volume ratio, OBV, VWAP distance, and calendar fields
- Skips expensive feature-selection ensemble
- Disables walk-forward CV
- Disables LightGBM
- Uses sequence length 20
- Uses regression labels
- Runs only CNN, LSTM, and Transformer

## Windows PowerShell

If the environment already exists:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_quick_ta_regression_deep.ps1
```

If the environment does not exist, the same command will call the one-click setup first.

## macOS / Linux

```bash
bash scripts/run_quick_ta_regression_deep.sh
```

## Direct Command

```bash
python main.py --config config/quick_ta_regression_deep.yaml
```

## Output

```text
experiments/quick_ta_regression_deep_h5_seq20/
```

Key files:

```text
reports/test_comparison.csv
reports/backtest_comparison.csv
reports/cnn/predictions.csv
reports/lstm/predictions.csv
reports/transformer/predictions.csv
```

In each predictions file:

- `prediction` is the model-predicted future return
- `target` is the realized future return label
- positive prediction means expected rise
- negative prediction means expected fall
