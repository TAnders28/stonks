"""Pull daily OHLCV via yfinance (keyless). Incremental merge into data/daily/<SYM>.json."""
from __future__ import annotations
import argparse, json, hashlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import yfinance as yf

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data" / "daily"
RAW = ROOT / "cache" / "raw"
CODE_VERSION = "stonk-data-0.1.0"


def _now_et() -> str:
    return datetime.now(ET).isoformat(timespec="seconds")


def _stamp() -> str:
    return datetime.now(ET).strftime("%Y%m%dT%H%M%S")


def fetch_daily(symbol: str, period: str = "5y") -> dict:
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
    if getattr(df.columns, "nlevels", 1) > 1:
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    bars = []
    for idx, row in df.iterrows():
        d = idx.date().isoformat() if hasattr(idx, "date") else str(idx)[:10]
        bars.append({
            "date": d,
            "begins_at": f"{d}T00:00:00Z",
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"]) if "Volume" in df.columns else 0.0,
        })
    return {
        "source": "yfinance",
        "symbol": symbol.upper(),
        "interval": "day",
        "adjustment_type": "all",  # auto_adjust=True => split+dividend
        "fetched_at_et": _now_et(),
        "source_last_bar_begins_at": bars[-1]["begins_at"] if bars else None,
        "bar_count": len(bars),
        "bars": bars,
        "code_version": CODE_VERSION,
        "notes": "Yahoo via yfinance auto_adjust=True (split+div). Free quotes often ~15m delayed. Survivorship bias: delisted names unavailable.",
    }


def merge_incremental(existing: dict | None, fresh: dict) -> dict:
    by_date = {}
    if existing and existing.get("bars"):
        for b in existing["bars"]:
            by_date[b["date"]] = b
    for b in fresh["bars"]:
        by_date[b["date"]] = b
    bars = [by_date[d] for d in sorted(by_date)]
    out = dict(fresh)
    out["bars"] = bars
    out["bar_count"] = len(bars)
    out["merged_from"] = existing.get("source") if existing else None
    out["source_last_bar_begins_at"] = bars[-1]["begins_at"] if bars else None
    return out


def save(symbol: str, payload: dict) -> Path:
    DAILY.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    stamp = _stamp()
    raw_path = RAW / f"yfinance_daily_{symbol}_{stamp}.json"
    raw_path.write_text(json.dumps(payload, indent=2))
    path = DAILY / f"{symbol}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY,HOOD,AXTI,SHOP,QURE")
    ap.add_argument("--period", default="5y")
    args = ap.parse_args()
    for sym in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        existing = None
        p = DAILY / f"{sym}.json"
        if p.exists():
            existing = json.loads(p.read_text())
        fresh = fetch_daily(sym, args.period)
        merged = merge_incremental(existing, fresh)
        path = save(sym, merged)
        print(f"{sym}: {merged['bar_count']} bars, last={merged['source_last_bar_begins_at']}, wrote {path}")


if __name__ == "__main__":
    main()
