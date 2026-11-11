"""Forward entry point: emit today's SMA signal from cached daily bars + freshness check."""
from __future__ import annotations
import argparse, json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from Backtest.Strategies.SmaCrossover import make_signal

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data" / "daily"
OUT = ROOT / "out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY,HOOD")
    ap.add_argument("--fast", type=int, default=20)
    ap.add_argument("--slow", type=int, default=50)
    ap.add_argument("--max-bar-age-days", type=int, default=4)  # weekends/holidays
    args = ap.parse_args()
    sig = make_signal(args.fast, args.slow)
    now = datetime.now(ET)
    picks = []
    for sym in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        path = DAILY / f"{sym}.json"
        data = json.loads(path.read_text())
        bars = data["bars"]
        last = bars[-1]
        last_dt = datetime.fromisoformat(last["date"]).replace(tzinfo=ET)
        age_days = (now.date() - last_dt.date()).days
        stale = age_days > args.max_bar_age_days
        desired = sig(bars, len(bars) - 1) if not stale else None
        picks.append({
            "symbol": sym,
            "as_of_bar_date": last["date"],
            "bar_age_days": age_days,
            "stale": stale,
            "signal": "LONG" if desired == 1 else ("FLAT" if desired == 0 else "NO_SIGNAL_STALE"),
            "last_close": last["close"],
            "data_source": data.get("source"),
            "fetched_at_et": data.get("fetched_at_et"),
            "source_last_bar_begins_at": data.get("source_last_bar_begins_at"),
            "rule": sig.name,
            "execution_note": "If trading, next session open — do not use this as a same-bar fill",
        })
    stamp = now.strftime("%Y%m%dT%H%M%S")
    out_dir = OUT / f"predict_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "picks.json").write_text(json.dumps({"run_at_et": now.isoformat(timespec="seconds"), "picks": picks}, indent=2))
    print(json.dumps(picks, indent=2))
    if any(p["stale"] for p in picks):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
