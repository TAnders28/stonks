"""Multi-asset daily engine: next-open fills, position caps, costs, no lookahead."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Callable
import math

CODE_VERSION = "stonk-data-0.2.0-highvol"


@dataclass
class CostModel:
    # bps each way (slippage + half-spread combined), by liquidity tier
    default_bps: float = 10.0
    symbol_bps: dict = field(default_factory=dict)
    commission: float = 0.0

    def bps_for(self, sym: str) -> float:
        return float(self.symbol_bps.get(sym, self.default_bps))

    def fill(self, open_px: float, side: str, sym: str) -> float:
        bps = self.bps_for(sym) / 10000.0
        return open_px * (1.0 + bps) if side == "buy" else open_px * (1.0 - bps)


@dataclass
class Trade:
    entry_date: str
    exit_date: str
    symbol: str
    side: str
    qty: float
    entry_price: float
    exit_price: float
    pnl: float


def max_drawdown(eqs: list[float]) -> float:
    peak = eqs[0] if eqs else 0.0
    mdd = 0.0
    for v in eqs:
        peak = max(peak, v)
        if peak > 0:
            mdd = min(mdd, (v - peak) / peak)
    return mdd


def sharpe(rets: list[float]) -> float:
    if len(rets) < 2:
        return float("nan")
    m = sum(rets) / len(rets)
    var = sum((r - m) ** 2 for r in rets) / (len(rets) - 1)
    sd = math.sqrt(var)
    return float("nan") if sd == 0 else m / sd * math.sqrt(252)


def sortino(rets: list[float]) -> float:
    if len(rets) < 2:
        return float("nan")
    m = sum(rets) / len(rets)
    downside = [r for r in rets if r < 0]
    if len(downside) < 1:
        return float("nan")
    var = sum(r ** 2 for r in downside) / len(downside)
    sd = math.sqrt(var)
    return float("nan") if sd == 0 else m / sd * math.sqrt(252)


def cagr(start: float, end: float, n_days: int) -> float:
    if start <= 0 or n_days <= 0 or end <= 0:
        return float("nan")
    years = n_days / 252.0
    return (end / start) ** (1 / years) - 1


def summarize(curve: list[dict], trades: list[Trade], start_equity: float, n_bars: int) -> dict:
    eqs = [e["equity"] for e in curve]
    rets = []
    for i in range(1, len(eqs)):
        if eqs[i - 1] > 0:
            rets.append(eqs[i] / eqs[i - 1] - 1)
    end = eqs[-1] if eqs else start_equity
    wins = [t for t in trades if t.pnl > 0]
    # exposure: fraction of days with any position
    exposed = sum(1 for e in curve if e.get("n_positions", 0) > 0)
    # turnover proxy: sum |delta weights| rough via trades notional / equity
    trade_notional = sum(abs(t.qty * t.entry_price) + abs(t.qty * t.exit_price) for t in trades)
    avg_eq = sum(eqs) / len(eqs) if eqs else start_equity
    return {
        "start_equity": start_equity,
        "end_equity": end,
        "total_return": end / start_equity - 1 if start_equity else float("nan"),
        "cagr": cagr(start_equity, end, n_bars),
        "max_drawdown": max_drawdown(eqs),
        "sharpe": sharpe(rets),
        "sortino": sortino(rets),
        "n_trades": len(trades),
        "win_rate": (len(wins) / len(trades)) if trades else float("nan"),
        "avg_trade_pnl": (sum(t.pnl for t in trades) / len(trades)) if trades else float("nan"),
        "exposure": exposed / len(curve) if curve else 0.0,
        "turnover_approx": (trade_notional / avg_eq / 2) if avg_eq else float("nan"),  # round-trips approx
    }


class Portfolio:
    """Long-only multi-name. Signals decide target weights at close; fills next open."""

    def __init__(self, start_equity: float, cost: CostModel, max_weight: float = 0.50):
        self.cash = start_equity
        self.start_equity = start_equity
        self.cost = cost
        self.max_weight = max_weight
        self.positions: dict[str, dict] = {}  # sym -> {qty, entry_px, entry_date}
        self.pending_targets: dict[str, float] | None = None  # weight targets
        self.curve: list[dict] = []
        self.trades: list[Trade] = []

    def equity_mark(self, closes: dict[str, float]) -> float:
        eq = self.cash
        for sym, pos in self.positions.items():
            eq += pos["qty"] * closes.get(sym, pos["entry_px"])
        return eq

    def execute_targets(self, date: str, opens: dict[str, float], targets: dict[str, float]):
        # clip targets to max_weight and renormalize if sum>1
        clipped = {s: min(self.max_weight, max(0.0, w)) for s, w in targets.items() if w > 0}
        ssum = sum(clipped.values())
        if ssum > 1.0:
            clipped = {s: w / ssum for s, w in clipped.items()}
        # mark equity at opens for sizing
        eq = self.cash
        for sym, pos in self.positions.items():
            eq += pos["qty"] * opens.get(sym, pos["entry_px"])
        if eq <= 0:
            return
        desired_qty = {}
        for sym, w in clipped.items():
            if sym not in opens or opens[sym] <= 0:
                continue
            px = self.cost.fill(opens[sym], "buy", sym)
            desired_qty[sym] = (w * eq) / px
        # exit names not in desired
        for sym in list(self.positions.keys()):
            if sym not in desired_qty:
                if sym not in opens:
                    continue
                px = self.cost.fill(opens[sym], "sell", sym)
                pos = self.positions.pop(sym)
                proceeds = pos["qty"] * px - self.cost.commission
                self.cash += proceeds
                self.trades.append(Trade(
                    entry_date=pos["entry_date"], exit_date=date, symbol=sym, side="long",
                    qty=pos["qty"], entry_price=pos["entry_px"], exit_price=px,
                    pnl=proceeds - pos["qty"] * pos["entry_px"],
                ))
        # rebalance existing / new
        for sym, dq in desired_qty.items():
            px_buy = self.cost.fill(opens[sym], "buy", sym)
            px_sell = self.cost.fill(opens[sym], "sell", sym)
            have = self.positions.get(sym, {"qty": 0.0, "entry_px": px_buy, "entry_date": date})
            delta = dq - have["qty"]
            if abs(delta) * opens[sym] < 1.0:  # ignore tiny
                continue
            if delta > 0:
                cost = delta * px_buy + self.cost.commission
                if cost > self.cash:
                    delta = max(0.0, (self.cash - self.cost.commission) / px_buy)
                    cost = delta * px_buy + self.cost.commission
                self.cash -= cost
                new_qty = have["qty"] + delta
                # average entry
                if have["qty"] <= 0:
                    entry_px, entry_date = px_buy, date
                else:
                    entry_px = (have["qty"] * have["entry_px"] + delta * px_buy) / new_qty
                    entry_date = have["entry_date"]
                self.positions[sym] = {"qty": new_qty, "entry_px": entry_px, "entry_date": entry_date}
            else:
                sell_qty = min(have["qty"], -delta)
                proceeds = sell_qty * px_sell - self.cost.commission
                self.cash += proceeds
                # partial exit: record trade for sold portion
                self.trades.append(Trade(
                    entry_date=have["entry_date"], exit_date=date, symbol=sym, side="long",
                    qty=sell_qty, entry_price=have["entry_px"], exit_price=px_sell,
                    pnl=proceeds - sell_qty * have["entry_px"],
                ))
                left = have["qty"] - sell_qty
                if left <= 1e-12:
                    self.positions.pop(sym, None)
                else:
                    self.positions[sym] = {"qty": left, "entry_px": have["entry_px"], "entry_date": have["entry_date"]}

    def mark(self, date: str, closes: dict[str, float]):
        eq = self.equity_mark(closes)
        self.curve.append({
            "date": date,
            "equity": eq,
            "n_positions": len(self.positions),
            "cash": self.cash,
        })
