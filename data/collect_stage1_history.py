from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage1_close_report import filter_stock_rows, load_sources, source_rows, to_template_row


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "sources.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect historical Taiwan OHLCV data using stage-1 official sources.")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Stage-1 source config")
    parser.add_argument("--output", default="data/raw/taiwan_daily_ohlcv.csv", help="Output CSV path")
    parser.add_argument("--include-esb-latest", action="store_true", help="Include ESB only when the requested date is today")
    parser.add_argument("--include-non-stock", action="store_true", help="Keep non four-digit products")
    return parser.parse_args()


def date_range(start: dt.date, end: dt.date) -> list[dt.date]:
    days: list[dt.date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += dt.timedelta(days=1)
    return days


def normalized_records(day: dt.date, market: str, label: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        values = to_template_row(market, row)
        if market == "listed":
            symbol, name = values[0], values[1]
            volume, amount = values[2], values[4]
            open_price, high, low, close = values[5], values[6], values[7], values[8]
        elif market == "mainboard":
            symbol, name = values[0], values[1]
            close, open_price, high, low = values[2], values[4], values[5], values[6]
            volume, amount = values[7], values[8]
        elif market == "esb":
            symbol, name = values[0], values[1]
            high, low, close = values[7], values[8], values[10]
            open_price = close
            volume, amount = values[13], None
        else:
            continue
        records.append(
            {
                "date": day.isoformat(),
                "symbol": str(symbol).strip(),
                "name": name,
                "market": label,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "adjusted_close": close,
                "volume": volume,
                "amount": amount,
            }
        )
    return records


def collect_history(start: dt.date, end: dt.date, config_path: Path, include_esb_latest: bool, include_non_stock: bool) -> pd.DataFrame:
    all_records: list[dict[str, Any]] = []
    today = dt.date.today()
    for day in date_range(start, end):
        print(f"Collecting {day.isoformat()}")
        sources = load_sources(config_path, day)
        for source in sources:
            if not source.historical and not (include_esb_latest and day == today):
                print(f"  {source.label}: skipped (latest-only source)")
                continue
            try:
                rows = source_rows(source)
            except Exception as exc:
                print(f"  {source.label}: skipped ({exc})")
                continue
            if not include_non_stock:
                rows = filter_stock_rows(rows)
            records = normalized_records(day, source.market, source.label, rows)
            all_records.extend(records)
            print(f"  {source.label}: {len(records)} rows")
    if not all_records:
        raise RuntimeError("No records collected. Check date range, network, and source availability.")
    frame = pd.DataFrame(all_records)
    numeric_columns = ["open", "high", "low", "close", "adjusted_close", "volume", "amount"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values(["date", "market", "symbol"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = collect_history(start, end, Path(args.config), args.include_esb_latest, args.include_non_stock)
    frame.to_csv(output_path, index=False)
    print(f"Wrote {len(frame)} rows to {output_path}")


if __name__ == "__main__":
    main()
