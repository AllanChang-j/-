from __future__ import annotations

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


EPS = 1e-12


def _wma(series: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window + 1, dtype=float)
    return series.rolling(window).apply(lambda values: np.dot(values, weights) / weights.sum(), raw=True)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x = x - x.mean()
    denominator = np.dot(x, x)

    def slope(values: np.ndarray) -> float:
        y = values - values.mean()
        return float(np.dot(x, y) / denominator)

    return series.rolling(window).apply(slope, raw=True)


def _rolling_poly_trend(series: pd.Series, window: int, degree: int = 2) -> pd.Series:
    x = np.arange(window, dtype=float)
    design = np.vander(x, degree + 1)
    weights = np.linalg.pinv(design)[0]

    def coef(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        return float(np.dot(weights, values))

    return series.rolling(window).apply(coef, raw=True)


def _rolling_regression_r2(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    centered_x = x - x.mean()
    ss_x = float(np.dot(centered_x, centered_x))

    def centered_xy(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        return float(np.dot(centered_x, values))

    numerator = series.rolling(window).apply(centered_xy, raw=True)
    rolling_sum = series.rolling(window).sum()
    rolling_sum_sq = (series**2).rolling(window).sum()
    ss_y = rolling_sum_sq - (rolling_sum**2 / window)
    return (numerator**2) / (ss_x * ss_y + EPS)


def _hurst_exponent(series: pd.Series, window: int) -> pd.Series:
    lags = np.arange(2, min(20, window // 2))
    if len(lags) < 2:
        return pd.Series(np.nan, index=series.index)

    log_tau_parts = []
    for lag in lags:
        tau = np.sqrt(series.diff(lag).rolling(window - lag).std())
        log_tau_parts.append(np.log(tau.where(tau > 0)))
    log_tau = pd.concat(log_tau_parts, axis=1)
    log_tau.columns = lags

    x = pd.Series(np.log(lags), index=lags, dtype=float)
    valid = log_tau.notna()
    count = valid.sum(axis=1)
    y_sum = log_tau.sum(axis=1, skipna=True)
    xy_sum = log_tau.mul(x, axis=1).sum(axis=1, skipna=True)
    x_sum = valid.mul(x, axis=1).sum(axis=1)
    x2_sum = valid.mul(x**2, axis=1).sum(axis=1)
    denominator = count * x2_sum - x_sum**2
    slope = (count * xy_sum - x_sum * y_sum) / (denominator + EPS)
    return (2 * slope).where(count >= 2)


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + EPS)
    return 100 - 100 / (1 + rs)


def _stochastic(close: pd.Series, high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    lowest = low.rolling(window).min()
    highest = high.rolling(window).max()
    return 100 * (close - lowest) / (highest - lowest + EPS)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1)
    return ranges.max(axis=1)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume.fillna(0)).cumsum()


def _cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    money_flow_multiplier = ((close - low) - (high - close)) / (high - low + EPS)
    money_flow_volume = money_flow_multiplier * volume
    return money_flow_volume.rolling(window).sum() / (volume.rolling(window).sum() + EPS)


def add_features_for_symbol(group: pd.DataFrame, windows: list[int], lag_windows: list[int]) -> pd.DataFrame:
    group = group.sort_values("date").copy()
    close = group["adjusted_close"].astype(float)
    raw_close = group["close"].astype(float)
    high = group["high"].astype(float)
    low = group["low"].astype(float)
    open_price = group["open"].astype(float)
    volume = group["volume"].astype(float)
    amount = group["amount"].astype(float) if "amount" in group else raw_close * volume

    group["daily_return"] = close.pct_change(fill_method=None)
    group["weekly_return"] = close.pct_change(5, fill_method=None)
    group["monthly_return"] = close.pct_change(20, fill_method=None)
    group["log_return"] = np.log(close / close.shift(1))
    group["cumulative_return_20"] = (1 + group["daily_return"]).rolling(20).apply(np.prod, raw=True) - 1
    group["intraday_return"] = raw_close / open_price - 1
    group["overnight_return"] = open_price / raw_close.shift(1) - 1
    group["high_low_range"] = (high - low) / (raw_close + EPS)
    group["close_open_range"] = (raw_close - open_price) / (open_price + EPS)

    true_range = _true_range(high, low, raw_close)
    group["true_range"] = true_range
    group["parkinson_volatility"] = np.sqrt((np.log(high / (low + EPS)) ** 2) / (4 * np.log(2)))
    group["obv"] = _obv(raw_close, volume)
    typical_price = (high + low + raw_close) / 3
    group["vwap"] = (typical_price * volume).cumsum() / (volume.cumsum() + EPS)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    group["macd"] = macd
    group["macd_signal"] = macd.ewm(span=9, adjust=False).mean()
    group["macd_hist"] = group["macd"] - group["macd_signal"]
    group["ppo"] = 100 * (ema_12 - ema_26) / (ema_26 + EPS)
    group["ppo_signal"] = group["ppo"].ewm(span=9, adjust=False).mean()

    hurst_window = max((window for window in windows if window >= 20), default=None)
    for window in windows:
        ma = close.rolling(window).mean()
        ema = close.ewm(span=window, adjust=False).mean()
        wma = _wma(close, window)
        group[f"ma_{window}"] = ma
        group[f"ema_{window}"] = ema
        group[f"wma_{window}"] = wma
        group[f"close_to_ma_{window}"] = close / (ma + EPS) - 1
        group[f"close_to_ema_{window}"] = close / (ema + EPS) - 1
        group[f"slope_{window}"] = close.diff(window) / window
        group[f"linreg_slope_{window}"] = _rolling_slope(close, window)
        group[f"poly_trend_{window}"] = _rolling_poly_trend(close, window)
        group[f"rolling_regression_r2_{window}"] = _rolling_regression_r2(close, window)
        if window == hurst_window:
            group[f"hurst_{window}"] = _hurst_exponent(close, window)

        group[f"rsi_{window}"] = _rsi(close, window)
        group[f"stochastic_{window}"] = _stochastic(close, high, low, window)
        group[f"roc_{window}"] = close.pct_change(window, fill_method=None)
        group[f"momentum_{window}"] = close - close.shift(window)

        group[f"atr_{window}"] = true_range.rolling(window).mean()
        group[f"historical_volatility_{window}"] = group["log_return"].rolling(window).std() * np.sqrt(252)
        rolling_std = close.rolling(window).std()
        bollinger_width = 4 * rolling_std / (ma + EPS)
        group[f"bollinger_width_{window}"] = bollinger_width
        atr = group[f"atr_{window}"]
        keltner_mid = ema
        group[f"keltner_width_{window}"] = 4 * atr / (keltner_mid + EPS)

        group[f"cmf_{window}"] = _cmf(high, low, raw_close, volume, window)
        group[f"volume_ma_{window}"] = volume.rolling(window).mean()
        group[f"volume_ratio_{window}"] = volume / (group[f"volume_ma_{window}"] + EPS)
        group[f"amount_ma_{window}"] = amount.rolling(window).mean()
        group[f"amount_ratio_{window}"] = amount / (group[f"amount_ma_{window}"] + EPS)

        group[f"rolling_mean_{window}"] = group["daily_return"].rolling(window).mean()
        group[f"rolling_std_{window}"] = group["daily_return"].rolling(window).std()
        group[f"rolling_median_{window}"] = group["daily_return"].rolling(window).median()
        group[f"rolling_skew_{window}"] = group["daily_return"].rolling(window).skew()
        group[f"rolling_kurtosis_{window}"] = group["daily_return"].rolling(window).kurt()
        group[f"rolling_quantile_25_{window}"] = group["daily_return"].rolling(window).quantile(0.25)
        group[f"rolling_quantile_75_{window}"] = group["daily_return"].rolling(window).quantile(0.75)

    for lag in lag_windows:
        group[f"return_lag_{lag}"] = group["daily_return"].shift(lag)
        group[f"volume_ratio_lag_{lag}"] = group.get("volume_ratio_20", volume / (volume.rolling(20).mean() + EPS)).shift(lag)
        group[f"close_lag_{lag}"] = close.shift(lag)

    group["weekday"] = group["date"].dt.weekday
    group["month"] = group["date"].dt.month
    group["quarter"] = group["date"].dt.quarter
    group["is_month_end"] = group["date"].dt.is_month_end.astype(int)
    group["is_month_start"] = group["date"].dt.is_month_start.astype(int)
    group["holiday"] = group["weekday"].isin([5, 6]).astype(int)

    return group


def build_technical_features(df: pd.DataFrame, windows: list[int], lag_windows: list[int]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for symbol, group in df.sort_values(["symbol", "date"]).groupby("symbol", sort=False):
        enriched_group = add_features_for_symbol(group.copy(), windows, lag_windows)
        if "symbol" not in enriched_group.columns:
            enriched_group["symbol"] = symbol
        parts.append(enriched_group)
    if not parts:
        raise ValueError("No symbol groups were available for feature engineering.")
    enriched = pd.concat(parts, axis=0, ignore_index=True)
    enriched = enriched.replace([np.inf, -np.inf], np.nan)
    return enriched.copy().reset_index(drop=True)


def candidate_feature_columns(df: pd.DataFrame) -> list[str]:
    non_features = {
        "date",
        "symbol",
        "name",
        "market",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "future_return",
        "target",
        "target_class",
    }
    return [column for column in df.columns if column not in non_features and pd.api.types.is_numeric_dtype(df[column])]
