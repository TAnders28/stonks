"""Write Graphtard-ready tidy CSVs + meta alongside a backtest run."""
from __future__ import annotations
import csv, json
from pathlib import Path
from dataclasses import asdict


def write_prices_csv(path: Path, symbols_bars: dict[str, list[dict]]):
    """columns: date,symbol,open,high,low,close,volume"""
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["date", "symbol", "open", "high", "low", "close", "volume"])
        w.writeheader()
        for sym, bars in symbols_bars.items():
            for b in bars:
                w.writerow({
                    "date": b["date"],
                    "symbol": sym,
                    "open": b["open"],
                    "high": b["high"],
                    "low": b["low"],
                    "close": b["close"],
                    "volume": b["volume"],
                })


def _drawdowns(equities: list[float]) -> list[float]:
    peak = equities[0] if equities else 0.0
    out = []
    for v in equities:
        peak = max(peak, v)
        out.append((v - peak) / peak if peak > 0 else 0.0)
    return out


def write_equity_csv(path: Path, series_curves: dict[str, list[dict]]):
    """columns: date,series,equity,drawdown_pct  (drawdown negative fraction)"""
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["date", "series", "equity", "drawdown_pct"])
        w.writeheader()
        for series, curve in series_curves.items():
            eqs = [r["equity"] for r in curve]
            dds = _drawdowns(eqs)
            for r, dd in zip(curve, dds):
                w.writerow({
                    "date": r["date"],
                    "series": series,
                    "equity": r["equity"],
                    "drawdown_pct": dd,
                })


def write_trades_csv(path: Path, symbol: str, trades: list):
    """columns: entry_date,exit_date,symbol,side,qty,entry_price,exit_price,pnl"""
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "entry_date", "exit_date", "symbol", "side", "qty", "entry_price", "exit_price", "pnl"
        ])
        w.writeheader()
        for t in trades:
            d = asdict(t) if hasattr(t, "__dataclass_fields__") else t
            w.writerow({
                "entry_date": d["entry_date"],
                "exit_date": d["exit_date"],
                "symbol": symbol,
                "side": d.get("side", "long"),
                "qty": d.get("shares", d.get("qty")),
                "entry_price": d.get("entry_px", d.get("entry_price")),
                "exit_price": d.get("exit_px", d.get("exit_price")),
                "pnl": d["pnl"],
            })


def write_meta(path: Path, meta: dict):
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
