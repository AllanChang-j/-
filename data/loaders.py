from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


COLUMN_ALIASES = {
    "date": ["date", "Date", "日期", "交易日"],
    "symbol": ["symbol", "ticker", "code", "證券代號", "代號"],
    "name": ["name", "證券名稱", "名稱"],
    "market": ["market", "市場"],
    "open": ["open", "Open", "開盤", "開盤價"],
    "high": ["high", "High", "最高", "最高價"],
    "low": ["low", "Low", "最低", "最低價"],
    "close": ["close", "Close", "收盤", "收盤價", "成交"],
    "adjusted_close": ["adjusted_close", "adj_close", "Adj Close", "Adjusted Close", "還原收盤價"],
    "volume": ["volume", "Volume", "成交股數", "成交量"],
    "amount": ["amount", "成交金額", "成交金額(元)"],
}

REQUIRED_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]


def _resolve_columns(columns: list[str]) -> dict[str, str]:
    normalized = {str(col).strip(): col for col in columns}
    mapping: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                mapping[normalized[alias]] = canonical
                break
    return mapping


def load_daily_csv(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Daily data file not found: {input_path}")
    df = pd.read_csv(input_path)
    return normalize_daily_schema(df)


def normalize_daily_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=_resolve_columns(list(df.columns))).copy()
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required daily columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str).str.strip()
    if "name" not in df.columns:
        df["name"] = df["symbol"]
    if "market" not in df.columns:
        df["market"] = "unknown"
    if "adjusted_close" not in df.columns:
        df["adjusted_close"] = df["close"]

    numeric_columns = [column for column in df.columns if column not in {"date", "symbol", "name", "market"}]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")
    return df.reset_index(drop=True)


def split_extra_feature_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    base = {"date", "symbol", "name", "market", "open", "high", "low", "close", "adjusted_close", "volume", "amount"}
    extra = [column for column in df.columns if column not in base]
    fundamentals = [column for column in extra if column.startswith(("fund_", "fundamental_"))]
    macro = [column for column in extra if column.startswith(("macro_", "tw_macro_", "global_macro_"))]
    other = [column for column in extra if column not in set(fundamentals + macro)]
    return {"fundamental": fundamentals, "macro": macro, "other": other}


def validate_no_duplicate_timestamps(df: pd.DataFrame) -> None:
    duplicated = df.duplicated(["symbol", "date"]).sum()
    if duplicated:
        raise ValueError(f"Found {duplicated} duplicate symbol/date rows")

