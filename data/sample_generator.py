from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_synthetic_taiwan_daily(path: str | Path, seed: int = 42, n_days: int = 520) -> Path:
    rng = np.random.default_rng(seed)
    symbols = ["2330", "2317", "2454", "2881", "1301", "2303"]
    names = ["台積電", "鴻海", "聯發科", "富邦金", "台塑", "聯電"]
    markets = ["上市"] * len(symbols)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    records = []

    for symbol, name, market in zip(symbols, names, markets):
        drift = rng.normal(0.00025, 0.00008)
        vol = rng.uniform(0.012, 0.028)
        close = rng.uniform(40, 650)
        shares = rng.integers(20_000_000, 150_000_000)
        for date in dates:
            shock = rng.normal(drift, vol)
            prev_close = close
            close = max(5.0, prev_close * np.exp(shock))
            intraday = abs(rng.normal(0, vol / 2))
            high = max(close, prev_close) * (1 + intraday)
            low = min(close, prev_close) * (1 - intraday)
            open_price = prev_close * (1 + rng.normal(0, vol / 3))
            volume = max(1000, int(shares * rng.lognormal(-4.2, 0.45)))
            amount = volume * close
            records.append(
                {
                    "date": date.date().isoformat(),
                    "symbol": symbol,
                    "name": name,
                    "market": market,
                    "open": round(open_price, 2),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "close": round(close, 2),
                    "adjusted_close": round(close, 2),
                    "volume": volume,
                    "amount": round(amount, 0),
                    "fund_pe": max(3, rng.normal(18, 4)),
                    "fund_pb": max(0.2, rng.normal(2.2, 0.6)),
                    "fund_roe": rng.normal(0.13, 0.04),
                    "macro_taiex_return": rng.normal(0.0002, 0.01),
                    "macro_usd_twd_return": rng.normal(0.0, 0.002),
                }
            )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_path, index=False)
    return output_path

