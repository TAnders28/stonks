"""Keyless earnings calendar via yfinance Ticker.get_earnings_dates (Yahoo scrape).

Writes data/earnings/<SYM>.json (past + next dates, with ET timestamps) and data/earnings/calendar.json.
Coverage: Yahoo returns ~8 years (roughly 2018/2019 onward) per name; ETFs return nothing.
Before a name's first covered date the calendar is UNKNOWN (backtests must not pretend to skip).

Usage: .venv\\Scripts\\python -m DataPull.earnings  [--symbols MRNA,TWST]
"""
from __future__ import annotations
import argparse, json, time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import yfinance as yf

from DataPull.refresh import universe_symbols, UNIVERSE

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[1]
EARN = ROOT / "data" / "earnings"
CODE_VERSION = "stonk-data-0.3.0-earnings"
NO_EARNINGS_BUCKETS = {"leveraged_etfs", "leveraged_inverse", "benchmarks", "vol_proxy"}
KNOWN_ETFS = {"XBI", "IBB", "URA", "GDX", "GDXJ", "SIL", "BITO", "IBIT", "TAN", "MSOS", "SPY", "QQQ", "IWM", "VTI",
              "TLT", "BIL", "HYG", "IEF", "^VIX", "UVXY", "SVXY", "VIXY"}


def session_of_effect(ts: datetime) -> tuple[str, str]:
    """Return (date, timing) where timing BMO if before 12:00 ET else AMC."""
    t = ts.astimezone(ET)
    return t.date().isoformat(), ("BMO" if t.hour < 12 else "AMC")


def pull(sym: str, limit: int = 100) -> dict:
    df = yf.Ticker(sym).get_earnings_dates(limit=limit)
    rows = []
    if df is not None and len(df):
        for ts, r in df.iterrows():
            d, timing = session_of_effect(ts.to_pydatetime())
            def num(k):
                try:
                    v = float(r.get(k)); return v if v == v else None
                except Exception:
                    return None
            rows.append({"earnings_at_et": ts.to_pydatetime().astimezone(ET).isoformat(timespec="minutes"),
                         "date": d, "timing": timing, "eps_estimate": num("EPS Estimate"),
                         "eps_reported": num("Reported EPS")})
    rows.sort(key=lambda x: x["earnings_at_et"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=None)
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()
    u = json.loads(Path(UNIVERSE).read_text())
    skip = set(KNOWN_ETFS)
    for b, v in u["buckets"].items():
        if b in NO_EARNINGS_BUCKETS:
            skip |= set(v["symbols"])
    syms = [s.strip() for s in args.symbols.split(",")] if args.symbols else universe_symbols(Path(UNIVERSE))
    EARN.mkdir(parents=True, exist_ok=True)
    fetched = datetime.now(ET)
    today = fetched.date().isoformat()
    cal = {"fetched_at_et": fetched.isoformat(timespec="seconds"), "source": "yfinance get_earnings_dates (keyless Yahoo)",
           "code_version": CODE_VERSION, "symbols": {}}
    for s in syms:
        if s in skip:
            cal["symbols"][s] = {"status": "etf_or_index_no_earnings"}; continue
        try:
            rows = pull(s)
            nxt = next((r for r in rows if r["date"] >= today and r["eps_reported"] is None), None)
            rec = {"symbol": s, "fetched_at_et": cal["fetched_at_et"], "source": cal["source"], "dates": rows,
                   "coverage_first": rows[0]["date"] if rows else None, "next": nxt}
            (EARN / f"{s.replace('^','')}.json").write_text(json.dumps(rec, indent=1))
            cal["symbols"][s] = {"status": "ok" if rows else "empty", "n": len(rows),
                                 "coverage_first": rec["coverage_first"], "next": nxt}
            print(f"{s}: n={len(rows)} first={rec['coverage_first']} next={nxt['date'] + ' ' + nxt['timing'] if nxt else None}", flush=True)
        except Exception as e:
            cal["symbols"][s] = {"status": "error", "error": repr(e)[:200]}
            print(f"{s}: ERR {e}", flush=True)
        time.sleep(args.sleep)
    (EARN / "calendar.json").write_text(json.dumps(cal, indent=1))
    ok = sum(1 for v in cal["symbols"].values() if v["status"] == "ok")
    print(f"EARNINGS ok={ok} of {len(syms)} -> {EARN/'calendar.json'}")


if __name__ == "__main__":
    main()
