"""Bulk daily OHLCV pull for universe symbols. Keyless. Writes data/daily/<SYM>.json + stamped raw."""
from __future__ import annotations
import argparse, json, time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import yfinance as yf

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data" / "daily"
RAW = ROOT / "cache" / "raw"
CODE_VERSION = "stonk-data-0.2.0-highvol"


def now_et():
    return datetime.now(ET).isoformat(timespec="seconds")


def stamp():
    return datetime.now(ET).strftime("%Y%m%dT%H%M%S")


def fetch_one(symbol: str, period: str = "max") -> dict | None:
    # ^VIX and similar
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
    if df is None or len(df) == 0:
        return None
    if getattr(df.columns, "nlevels", 1) > 1:
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    bars = []
    for idx, row in df.iterrows():
        d = idx.date().isoformat() if hasattr(idx, "date") else str(idx)[:10]
        o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
        if not (o > 0 and h > 0 and l > 0 and c > 0):
            continue
        bars.append({
            "date": d,
            "begins_at": f"{d}T00:00:00Z",
            "open": o, "high": h, "low": l, "close": c,
            "volume": float(row["Volume"]) if "Volume" in df.columns else 0.0,
        })
    if len(bars) < 30:
        return None
    return {
        "source": "yfinance",
        "symbol": symbol,
        "interval": "day",
        "adjustment_type": "all",
        "fetched_at_et": now_et(),
        "source_last_bar_begins_at": bars[-1]["begins_at"],
        "bar_count": len(bars),
        "bars": bars,
        "code_version": CODE_VERSION,
        "notes": "auto_adjust=True (split+div). Survivorship: delisted often missing.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default=str(ROOT / "data/universe/highvol_2026-11-08.json"))
    ap.add_argument("--period", default="max")
    ap.add_argument("--sleep", type=float, default=0.15)
    args = ap.parse_args()
    u = json.loads(Path(args.universe).read_text())
    symbols = []
    for b in u["buckets"].values():
        symbols.extend(b["symbols"])
    symbols.extend(u.get("survivorship_attempts", {}).get("included_if_yahoo_serves", []))
    # unique preserve order
    seen = set(); syms = []
    for s in symbols:
        if s not in seen:
            seen.add(s); syms.append(s)
    DAILY.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    ok, fail = [], []
    summary = {"fetched_at_et": now_et(), "period": args.period, "ok": {}, "fail": {}}
    for i, sym in enumerate(syms):
        try:
            payload = fetch_one(sym, args.period)
            if not payload:
                fail.append(sym); summary["fail"][sym] = "empty_or_short"
                print(f"[{i+1}/{len(syms)}] FAIL {sym}")
            else:
                # safe filename for ^VIX
                fname = sym.replace("^", "")
                path = DAILY / f"{fname}.json"
                path.write_text(json.dumps(payload))
                (RAW / f"yfinance_daily_{fname}_{stamp()}.json").write_text(json.dumps({
                    "symbol": sym, "bar_count": payload["bar_count"],
                    "first": payload["bars"][0]["date"], "last": payload["bars"][-1]["date"],
                    "fetched_at_et": payload["fetched_at_et"],
                }, indent=2))
                ok.append(sym)
                summary["ok"][sym] = {
                    "bars": payload["bar_count"],
                    "first": payload["bars"][0]["date"],
                    "last": payload["bars"][-1]["date"],
                    "file": str(path),
                }
                print(f"[{i+1}/{len(syms)}] OK {sym} {payload['bar_count']} {payload['bars'][0]['date']}->{payload['bars'][-1]['date']}")
        except Exception as e:
            fail.append(sym); summary["fail"][sym] = str(e)
            print(f"[{i+1}/{len(syms)}] ERR {sym} {e}")
        time.sleep(args.sleep)
    out = ROOT / "out" / f"universe_pull_{stamp()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary["n_ok"] = len(ok); summary["n_fail"] = len(fail)
    out.write_text(json.dumps(summary, indent=2))
    print("SUMMARY", summary["n_ok"], "ok", summary["n_fail"], "fail", "->", out)


if __name__ == "__main__":
    main()
