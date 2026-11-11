"""Leak-free daily bar engine: signal on bar t close, execute at bar t+1 open. Costs modeled."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Callable
import math

CODE_VERSION = "stonk-data-0.1.0"


@dataclass
class CostModel:
    commission_per_trade: float = 0.0  # $ flat per fill
    slippage_bps: float = 5.0  # basis points vs open
    spread_bps: float = 5.0  # half-spread approx charged each way

    def fill_price(self, open_px: float, side: str) -> float:
        # side: buy -> pay up; sell -> receive less
        bps = (self.slippage_bps + self.spread_bps) / 10000.0
        if side == "buy":
            return open_px * (1.0 + bps)
        return open_px * (1.0 - bps)


@dataclass
class Trade:
    entry_date: str
    exit_date: str
    side: str
    entry_px: float
    exit_px: float
    shares: float
    pnl: float
    ret: float
    bars_held: int


@dataclass
class Book:
    cash: float
    equity_curve: list = field(default_factory=list)  # {date, equity, position}
    trades: list = field(default_factory=list)
    position: float = 0.0  # shares
    entry_px: float = 0.0
    entry_date: str = ""
    entry_bar_i: int = 0
    pending: str | None = None  # 'buy' or 'sell' to execute next open


def _max_drawdown(curve: list[float]) -> float:
    peak = curve[0] if curve else 0.0
    mdd = 0.0
    for v in curve:
        peak = max(peak, v)
        if peak > 0:
            mdd = min(mdd, (v - peak) / peak)
    return mdd  # negative


def _sharpe(daily_rets: list[float], rf: float = 0.0) -> float:
    if len(daily_rets) < 2:
        return float("nan")
    mean = sum(daily_rets) / len(daily_rets)
    var = sum((r - mean) ** 2 for r in daily_rets) / (len(daily_rets) - 1)
    std = math.sqrt(var)
    if std == 0:
        return float("nan")
    return (mean - rf / 252.0) / std * math.sqrt(252.0)


def _cagr(start: float, end: float, n_days: int) -> float:
    if start <= 0 or n_days <= 0:
        return float("nan")
    years = n_days / 252.0
    return (end / start) ** (1.0 / years) - 1.0


def summarize(book: Book, start_equity: float, n_bars: int) -> dict:
    eqs = [e["equity"] for e in book.equity_curve]
    daily_rets = []
    for i in range(1, len(eqs)):
        if eqs[i - 1] > 0:
            daily_rets.append(eqs[i] / eqs[i - 1] - 1.0)
    wins = [t for t in book.trades if t.pnl > 0]
    losses = [t for t in book.trades if t.pnl <= 0]
    end = eqs[-1] if eqs else start_equity
    return {
        "start_equity": start_equity,
        "end_equity": end,
        "total_return": end / start_equity - 1.0 if start_equity else float("nan"),
        "cagr": _cagr(start_equity, end, n_bars),
        "max_drawdown": _max_drawdown(eqs),
        "sharpe": _sharpe(daily_rets),
        "n_trades": len(book.trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (len(wins) / len(book.trades)) if book.trades else float("nan"),
        "avg_trade_ret": (sum(t.ret for t in book.trades) / len(book.trades)) if book.trades else float("nan"),
    }


def run_long_only(
    bars: list[dict],
    signal_fn: Callable[[list[dict], int], int],
    *,
    start_equity: float = 10_000.0,
    cost: CostModel | None = None,
    start_i: int = 0,
    end_i: int | None = None,
) -> Book:
    """
    signal_fn(bars, i) -> desired position after seeing bars[0..i] inclusive (close of i).
      1 = want long, 0 = want flat. Execution at next bar's open.
    """
    cost = cost or CostModel()
    end_i = end_i if end_i is not None else len(bars) - 1
    book = Book(cash=start_equity)
    prev_equity = start_equity

    for i in range(start_i, end_i + 1):
        bar = bars[i]
        # 1) execute pending at today's open
        if book.pending == "buy" and book.position == 0:
            px = cost.fill_price(bar["open"], "buy")
            # all-in
            spendable = book.cash - cost.commission_per_trade
            if spendable > 0 and px > 0:
                shares = spendable / px
                book.cash -= shares * px + cost.commission_per_trade
                book.position = shares
                book.entry_px = px
                book.entry_date = bar["date"]
                book.entry_bar_i = i
        elif book.pending == "sell" and book.position > 0:
            px = cost.fill_price(bar["open"], "sell")
            proceeds = book.position * px - cost.commission_per_trade
            pnl = proceeds - (book.position * book.entry_px)
            ret = (px / book.entry_px) - 1.0
            book.trades.append(Trade(
                entry_date=book.entry_date,
                exit_date=bar["date"],
                side="long",
                entry_px=book.entry_px,
                exit_px=px,
                shares=book.position,
                pnl=pnl,
                ret=ret,
                bars_held=i - book.entry_bar_i,
            ))
            book.cash += proceeds
            book.position = 0.0
            book.entry_px = 0.0
            book.entry_date = ""
        book.pending = None

        # mark-to-market on close
        mtm = book.cash + book.position * bar["close"]
        book.equity_curve.append({"date": bar["date"], "equity": mtm, "position": book.position, "close": bar["close"]})

        # 2) signal using bars up to i only (no look-ahead)
        desired = signal_fn(bars, i)
        desired = 1 if desired > 0 else 0
        have = 1 if book.position > 0 else 0
        if desired > have:
            book.pending = "buy"
        elif desired < have:
            book.pending = "sell"

        prev_equity = mtm

    # flatten at last close mark only for reporting — do not invent a fill beyond end
    return book


def buy_and_hold(bars: list[dict], start_equity: float = 10_000.0, cost: CostModel | None = None,
                 start_i: int = 0, end_i: int | None = None) -> Book:
    cost = cost or CostModel()
    end_i = end_i if end_i is not None else len(bars) - 1

    def always_long(bars_, i):
        return 1 if i >= start_i else 0

    # Force buy on first bar via pending trick: start flat, signal long from start_i-1 conceptually
    # Simpler: manually buy at start_i open then hold
    book = Book(cash=start_equity)
    first = True
    for i in range(start_i, end_i + 1):
        bar = bars[i]
        if first:
            px = cost.fill_price(bar["open"], "buy")
            spendable = book.cash - cost.commission_per_trade
            shares = spendable / px
            book.cash -= shares * px + cost.commission_per_trade
            book.position = shares
            book.entry_px = px
            book.entry_date = bar["date"]
            book.entry_bar_i = i
            first = False
        mtm = book.cash + book.position * bar["close"]
        book.equity_curve.append({"date": bar["date"], "equity": mtm, "position": book.position, "close": bar["close"]})
    # close trade at last open? report unrealized as exit at last close for summary trade
    if book.position > 0:
        last = bars[end_i]
        px = last["close"]  # mark exit for trade log (no extra cost beyond initial)
        pnl = book.position * px - book.position * book.entry_px
        book.trades.append(Trade(
            entry_date=book.entry_date, exit_date=last["date"], side="long",
            entry_px=book.entry_px, exit_px=px, shares=book.position,
            pnl=pnl, ret=px / book.entry_px - 1.0, bars_held=end_i - book.entry_bar_i,
        ))
    return book
