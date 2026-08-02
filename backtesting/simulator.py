from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 1_000_000
    commission_rate: float = 0.001425
    tax_rate: float = 0.003
    slippage_bps: float = 5
    position_size: float = 0.1
    max_positions: int = 5
    stop_loss: float = 0.08
    take_profit: float = 0.15
    long_threshold: float = 0.55
    short_threshold: float = 0.45
    allow_short: bool = False


def _annualized_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    return float((1 + total_return) ** (1 / years) - 1)


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min())


def _profit_factor(trade_returns: pd.Series) -> float:
    gains = trade_returns[trade_returns > 0].sum()
    losses = -trade_returns[trade_returns < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def backtest_predictions(prediction_frame: pd.DataFrame, config: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame = prediction_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if "score" not in frame.columns:
        frame["score"] = frame["signal"].astype(float)
    frame = frame.dropna(subset=["future_return"]).sort_values(["date", "score"], ascending=[True, False])

    cash = float(config.initial_cash)
    equity_records = []
    trades = []
    slippage = config.slippage_bps / 10_000
    cost_in = config.commission_rate + slippage
    cost_out = config.commission_rate + config.tax_rate + slippage

    for date, daily in frame.groupby("date", sort=True):
        long_candidates = daily[daily["signal"] > 0].head(config.max_positions)
        short_candidates = daily[daily["signal"] < 0].head(config.max_positions) if config.allow_short else daily.iloc[0:0]
        candidates = pd.concat([long_candidates, short_candidates], axis=0)
        if candidates.empty:
            equity_records.append({"date": date, "equity": cash, "daily_return": 0.0, "positions": 0})
            continue

        start_cash = cash
        allocation = min(cash * config.position_size, cash / max(len(candidates), 1))
        day_pnl = 0.0
        for _, row in candidates.iterrows():
            direction = 1 if row["signal"] > 0 else -1
            gross_return = float(row["future_return"]) * direction
            gross_return = max(gross_return, -config.stop_loss)
            gross_return = min(gross_return, config.take_profit)
            net_return = gross_return - cost_in - cost_out
            pnl = allocation * net_return
            day_pnl += pnl
            trades.append(
                {
                    "entry_date": date,
                    "symbol": row["symbol"],
                    "name": row.get("name", row["symbol"]),
                    "direction": "long" if direction > 0 else "short",
                    "score": row.get("score"),
                    "future_return": row["future_return"],
                    "net_return": net_return,
                    "pnl": pnl,
                    "capital": allocation,
                }
            )
        cash += day_pnl
        equity_records.append(
            {
                "date": date,
                "equity": cash,
                "daily_return": cash / start_cash - 1 if start_cash else 0.0,
                "positions": len(candidates),
            }
        )

    equity = pd.DataFrame(equity_records).set_index("date") if equity_records else pd.DataFrame(columns=["equity", "daily_return", "positions"])
    trades_df = pd.DataFrame(trades)
    if equity.empty:
        metrics = {"cumulative_return": 0.0, "trade_count": 0}
        return equity.reset_index(), trades_df, metrics

    returns = equity["daily_return"].fillna(0)
    downside = returns[returns < 0]
    volatility = returns.std(ddof=1) * np.sqrt(252) if len(returns) > 1 else 0.0
    sharpe = returns.mean() / (returns.std(ddof=1) + 1e-12) * np.sqrt(252) if len(returns) > 1 else 0.0
    sortino = returns.mean() / (downside.std(ddof=1) + 1e-12) * np.sqrt(252) if len(downside) > 1 else 0.0
    max_drawdown = _max_drawdown(equity["equity"])
    calmar = _annualized_return(equity["equity"]) / abs(max_drawdown) if max_drawdown < 0 else 0.0
    trade_returns = trades_df["net_return"] if not trades_df.empty else pd.Series(dtype=float)

    metrics = {
        "cumulative_return": float(equity["equity"].iloc[-1] / config.initial_cash - 1),
        "annual_return": _annualized_return(equity["equity"]),
        "volatility": float(volatility),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "calmar_ratio": float(calmar),
        "maximum_drawdown": float(max_drawdown),
        "win_rate": float((trade_returns > 0).mean()) if len(trade_returns) else 0.0,
        "profit_factor": _profit_factor(trade_returns),
        "average_holding_days": float("nan"),
        "trade_count": int(len(trades_df)),
        "final_equity": float(equity["equity"].iloc[-1]),
    }
    return equity.reset_index(), trades_df, metrics
