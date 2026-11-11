"""Live-ish price snapshot + historical display (keyless yfinance; quotes ~15 min delayed).

Writes out/snap_<stamp>/:
  <base>_snapshot.csv   symbol,last,prev_close,chg_pct,source_time_et,fetched_at_et,age_seconds,stale
  <base>_prices.csv     tidy intraday 5m bars for the latest session (date ISO w/ offset,symbol,o,h,l,c,v)
  <base>_daily.csv      tidy daily bars, last 126 sessions, from data/daily (date,symbol,o,h,l,c,v)
  <base>_draft.png      small multiples: 6-month daily close + latest quote marker (quick draft; Graphtard owns style)
Exit 2 if any quote is stale (fail closed) -- the files are still written, with stale=True rows.

Usage: .venv\\Scripts\\python -m DataPull.snapshot --symbols SPY,HOOD,AXTI,SHOP,QURE
"""
from __future__ import annotations
import argparse, csv, json, math, sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import yfinance as yf

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data" / "daily"


def intraday(sym: str):
    h = yf.Ticker(sym).history(period="1d", interval="5m", auto_adjust=False, prepost=False)
    rows = []
    for ts, r in h.iterrows():
        t = ts.to_pydatetime().astimezone(ET)
        rows.append({"date": t.isoformat(timespec="minutes"), "symbol": sym, "open": float(r["Open"]), "high": float(r["High"]),
                     "low": float(r["Low"]), "close": float(r["Close"]), "volume": float(r["Volume"])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY,HOOD,AXTI,SHOP,QURE")
    ap.add_argument("--max-stale-min", type=float, default=30.0)
    ap.add_argument("--slug", default="price-snapshot")
    args = ap.parse_args()
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    fetched = datetime.now(ET)
    run = ROOT / "out" / f"snap_{fetched.strftime('%Y%m%dT%H%M%S')}"
    run.mkdir(parents=True, exist_ok=False)
    base = f"{fetched.strftime('%Y-%m-%d')}_{args.slug}"
    snap, bars_all, daily_all = [], [], []
    for s in syms:
        rows = []
        try:
            rows = intraday(s)
        except Exception as e:
            print(f"{s}: intraday ERR {e}")
        bars_all += rows
        p = DAILY / f"{s.replace('^','')}.json"
        dbars = json.loads(p.read_text())["bars"][-126:] if p.exists() else []
        for b in dbars:
            daily_all.append({"date": b["date"], "symbol": s, **{k: b[k] for k in ("open", "high", "low", "close", "volume")}})
        last = rows[-1]["close"] if rows else None
        src = datetime.fromisoformat(rows[-1]["date"]) if rows else None
        # 5m bar timestamp is bar START; bar end = +5 min
        age = (fetched - src).total_seconds() - 300 if src else None
        prev = dbars[-1]["close"] if dbars else None
        if dbars and src and dbars[-1]["date"] == src.date().isoformat() and len(dbars) > 1:
            prev = dbars[-2]["close"]
        stale = (age is None) or (age > args.max_stale_min * 60)
        snap.append({"symbol": s, "last": last, "prev_close": prev,
                     "chg_pct": (last / prev - 1) * 100 if last and prev else None,
                     "source_time_et": src.isoformat(timespec="minutes") if src else None,
                     "fetched_at_et": fetched.isoformat(timespec="seconds"),
                     "age_seconds": round(age) if age is not None else None, "stale": stale})
    def wcsv(path, rows, cols):
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow({k: r.get(k) for k in cols}) for r in rows]
    wcsv(run / f"{base}_snapshot.csv", snap, ["symbol", "last", "prev_close", "chg_pct", "source_time_et", "fetched_at_et", "age_seconds", "stale"])
    pc = ["date", "symbol", "open", "high", "low", "close", "volume"]
    wcsv(run / f"{base}_prices.csv", bars_all, pc)
    wcsv(run / f"{base}_daily.csv", daily_all, pc)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        n = len(syms); cols = min(3, n); rws = math.ceil(n / cols)
        fig, axes = plt.subplots(rws, cols, figsize=(5 * cols, 3 * rws), squeeze=False)
        for k, s in enumerate(syms):
            ax = axes[k // cols][k % cols]
            d = [r for r in daily_all if r["symbol"] == s]
            ax.plot([datetime.fromisoformat(r["date"]) for r in d], [r["close"] for r in d], lw=1.2)
            q = next(x for x in snap if x["symbol"] == s)
            if q["last"]:
                ax.axhline(q["last"], color="red" if q["stale"] else "green", lw=0.8, ls="--")
            ax.set_title(f"{s} last {q['last']:.2f} ({q['chg_pct']:+.1f}%)" if q["last"] and q["chg_pct"] is not None else s, fontsize=9)
            ax.tick_params(labelsize=7)
        for k in range(n, rws * cols):
            axes[k // cols][k % cols].axis("off")
        fig.suptitle(f"Snapshot {fetched.strftime('%Y-%m-%d %H:%M')} ET (Yahoo, ~15m delayed; DRAFT)", fontsize=10)
        fig.tight_layout(); fig.savefig(run / f"{base}_draft.png", dpi=110); plt.close(fig)
    except Exception as e:
        print("chart ERR", e)
    meta = {"base_name": base, "fetched_at_et": fetched.isoformat(timespec="seconds"), "source": "yfinance (keyless Yahoo, ~15m delayed)",
            "max_stale_min": args.max_stale_min, "any_stale": any(x["stale"] for x in snap), "symbols": syms,
            "time_note": "source_time_et = start of last 5m bar (exchange time, ET); fetched_at_et = our clock"}
    (run / f"{base}_meta.json").write_text(json.dumps(meta, indent=2))
    for x in snap:
        print(json.dumps(x))
    print("SNAPSHOT", run)
    sys.exit(2 if meta["any_stale"] else 0)


if __name__ == "__main__":
    main()
