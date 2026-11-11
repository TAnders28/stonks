"""Near-live quotes via yfinance. Fail-closed on stale quotes."""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import yfinance as yf

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[1]
MAX_STALE_SECONDS = 30 * 60  # fail closed if quote older than 30m (weekend/holiday: report stale)


def quote(symbol: str, max_stale_s: int = MAX_STALE_SECONDS) -> dict:
    fetched = datetime.now(ET)
    t = yf.Ticker(symbol)
    fi = t.fast_info
    last = float(getattr(fi, "last_price", None) or getattr(fi, "lastPrice", None) or 0) or None
    prev = float(getattr(fi, "previous_close", None) or getattr(fi, "previousClose", None) or 0) or None
    # yfinance fast_info lacks a precise venue timestamp; treat as fetch-time with stale warning outside RTH
    # Use history 1m last bar time when available
    source_ts = None
    try:
        h = t.history(period="1d", interval="1m")
        if len(h):
            source_ts = h.index[-1].to_pydatetime()
            if source_ts.tzinfo is None:
                source_ts = source_ts.replace(tzinfo=ET)
            else:
                source_ts = source_ts.astimezone(ET)
    except Exception:
        source_ts = None
    age_s = None
    stale = False
    if source_ts is not None:
        age_s = (fetched - source_ts).total_seconds()
        stale = age_s > max_stale_s
    else:
        stale = True  # fail closed: no source timestamp
        age_s = None
    return {
        "symbol": symbol.upper(),
        "last_price": last,
        "previous_close": prev,
        "fetched_at_et": fetched.isoformat(timespec="seconds"),
        "source_timestamp_et": source_ts.isoformat(timespec="seconds") if source_ts else None,
        "age_seconds": age_s,
        "stale": stale,
        "max_stale_seconds": max_stale_s,
        "source": "yfinance",
        "usable_for_live_decision": (not stale) and last is not None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY,HOOD")
    ap.add_argument("--max-stale-s", type=int, default=MAX_STALE_SECONDS)
    args = ap.parse_args()
    out = [quote(s.strip(), args.max_stale_s) for s in args.symbols.split(",") if s.strip()]
    print(json.dumps(out, indent=2))
    if any(q["stale"] for q in out):
        raise SystemExit(2)  # fail closed


if __name__ == "__main__":
    main()
