"""Incremental daily-bar refresh for the high-vol universe + benchmarks/regime inputs. Keyless (yfinance).

- Pulls only recent bars (overlap window) per symbol and merges into data/daily/<SYM>.json.
- If the overlap closes disagree (> --adj-tol), Yahoo re-adjusted history (split/dividend) ->
  refetch full history for that symbol and replace it.
- Never stores a partial (in-progress) session: bars dated after the expected last completed
  session (16:30 ET rule, NYSE holidays) are dropped.
- Records source time (last bar date / Yahoo bar timestamp) separately from fetch time.
- Writes data/health/health_<stamp>.json + data/health/latest.md and appends data/health/refresh_log.jsonl.
- FAIL CLOSED: exit code 2 if any *required* symbol is stale (last bar < expected session) or failed.

Usage:  .venv\\Scripts\\python -m DataPull.refresh            (whole universe)
        .venv\\Scripts\\python -m DataPull.refresh --symbols SPY,HOOD
"""
from __future__ import annotations
import argparse, json, math, sys, time
from datetime import datetime, timedelta, date
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from DataPull.market_calendar import expected_last_complete_session, is_session, prev_session

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data" / "daily"
RAW = ROOT / "cache" / "raw"
HEALTH = ROOT / "data" / "health"
UNIVERSE = ROOT / "data" / "universe" / "highvol_2026-11-08.json"
CODE_VERSION = "stonk-data-0.3.0-refresh"
EXTRA = ["SPY", "QQQ", "IWM", "VTI", "TLT", "BIL", "HYG", "IEF", "^VIX"]
DEAD_AFTER_SESSIONS = 30  # last bar older than this many sessions -> treated as delisted/dead (not required)


def now_et() -> datetime:
    return datetime.now(ET)


def fname(sym: str) -> str:
    return sym.replace("^", "")


def universe_symbols(path: Path) -> list[str]:
    u = json.loads(path.read_text())
    out = []
    for b in u["buckets"].values():
        out += b["symbols"]
    out += EXTRA
    seen, res = set(), []
    for s in out:
        if s not in seen:
            seen.add(s); res.append(s)
    return res


def df_to_bars(df: pd.DataFrame) -> list[dict]:
    bars = []
    if df is None or len(df) == 0:
        return bars
    if getattr(df.columns, "nlevels", 1) > 1:
        df = df.copy(); df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    for idx, row in df.iterrows():
        try:
            o, h, l, c = (float(row[k]) for k in ("Open", "High", "Low", "Close"))
        except Exception:
            continue
        if not all(map(math.isfinite, (o, h, l, c))) or min(o, h, l, c) <= 0:
            continue
        d = idx.date().isoformat() if hasattr(idx, "date") else str(idx)[:10]
        v = float(row["Volume"]) if "Volume" in df.columns and math.isfinite(float(row["Volume"])) else 0.0
        bars.append({"date": d, "begins_at": f"{d}T00:00:00Z", "open": o, "high": h, "low": l, "close": c, "volume": v})
    return bars


def fetch(sym: str, start: str | None = None, period: str | None = None) -> list[dict]:
    kw = dict(interval="1d", auto_adjust=True, progress=False, threads=False)
    if start:
        df = yf.download(sym, start=start, **kw)
    else:
        df = yf.download(sym, period=period or "max", **kw)
    return df_to_bars(df)


def check_bars(bars: list[dict]) -> dict:
    issues = {"dup_dates": 0, "non_monotonic": 0, "ohlc_inconsistent": 0, "abs_ret_gt_50pct": 0, "zero_volume_last20": 0}
    prev = None
    for i, b in enumerate(bars):
        if prev is not None:
            if b["date"] == prev["date"]:
                issues["dup_dates"] += 1
            elif b["date"] < prev["date"]:
                issues["non_monotonic"] += 1
            if prev["close"] > 0 and abs(b["close"] / prev["close"] - 1) > 0.5:
                issues["abs_ret_gt_50pct"] += 1
        tol = 1e-6 * max(b["high"], 1)
        if b["high"] + tol < max(b["open"], b["close"]) or b["low"] - tol > min(b["open"], b["close"]):
            issues["ohlc_inconsistent"] += 1
        prev = b
    issues["zero_volume_last20"] = sum(1 for b in bars[-20:] if b.get("volume", 0) == 0)
    return issues


def sessions_between(d0: date, d1: date) -> int:
    """number of sessions strictly after d0 up to and including d1"""
    n, d = 0, d0
    while d < d1:
        d += timedelta(days=1)
        if is_session(d):
            n += 1
    return n


def refresh_symbol(sym: str, expected: date, overlap_days: int, adj_tol: float) -> dict:
    p = DAILY / f"{fname(sym)}.json"
    existing = json.loads(p.read_text()) if p.exists() else None
    old_bars = existing.get("bars", []) if existing else []
    old_source = existing.get("source") if existing else None
    action = "incremental"
    fetched_at = now_et()
    if not old_bars or old_source not in ("yfinance",):
        new = fetch(sym, period="max"); action = "full" if not old_bars else f"full(replace {old_source})"
    else:
        start = (date.fromisoformat(old_bars[-1]["date"]) - timedelta(days=overlap_days)).isoformat()
        new = fetch(sym, start=start)
    new = [b for b in new if b["date"] <= expected.isoformat()]  # drop partial session
    readjusted = False
    if action == "incremental" and new:
        old_by = {b["date"]: b for b in old_bars}
        diffs = [abs(b["close"] / old_by[b["date"]]["close"] - 1) for b in new if b["date"] in old_by]
        if diffs and max(diffs) > adj_tol:
            readjusted = True; action = f"full(readjust maxdiff={max(diffs):.4f})"
            new = [b for b in fetch(sym, period="max") if b["date"] <= expected.isoformat()]
    if action.startswith("full"):
        bars = new if new else old_bars
    else:
        by = {b["date"]: b for b in old_bars}
        for b in new:
            by[b["date"]] = b
        bars = [by[d] for d in sorted(by)]
    added = len(bars) - len(old_bars)
    if not bars:
        return {"symbol": sym, "ok": False, "error": "no data", "action": action}
    payload = {
        "source": "yfinance", "symbol": sym, "interval": "day", "adjustment_type": "all",
        "fetched_at_et": fetched_at.isoformat(timespec="seconds"),
        "source_last_bar_date": bars[-1]["date"],
        "source_last_bar_begins_at": bars[-1]["begins_at"],
        "source_time_note": "Yahoo daily bar date (exchange session, ET). fetched_at_et is our clock.",
        "bar_count": len(bars), "bars": bars, "code_version": CODE_VERSION,
        "notes": "auto_adjust=True (split+div). Partial sessions dropped. Survivorship: delisted names often missing.",
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload))
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / f"yfinance_refresh_{fname(sym)}_{fetched_at.strftime('%Y%m%dT%H%M%S')}.json").write_text(json.dumps({
        "symbol": sym, "action": action, "fetched_at_et": payload["fetched_at_et"],
        "n_new_bars_fetched": len(new), "new_bars": new[-15:], "last": bars[-1]["date"]}))
    last = date.fromisoformat(bars[-1]["date"])
    lag = sessions_between(last, expected)
    return {"symbol": sym, "ok": True, "action": action, "readjusted": readjusted, "bars": len(bars),
            "added": added, "first": bars[0]["date"], "last": bars[-1]["date"], "lag_sessions": lag,
            "fetched_at_et": payload["fetched_at_et"], "last_close": bars[-1]["close"], "issues": check_bars(bars)}


def write_health(rows: list[dict], expected: date, started: datetime, required: set[str]) -> tuple[Path, int]:
    HEALTH.mkdir(parents=True, exist_ok=True)
    stale, dead, failed = [], [], []
    for r in rows:
        if not r.get("ok"):
            failed.append(r["symbol"]); r["status"] = "FAILED"; continue
        if r["lag_sessions"] > DEAD_AFTER_SESSIONS:
            r["status"] = "DEAD/DELISTED"; dead.append(r["symbol"])
        elif r["lag_sessions"] > 0:
            r["status"] = "STALE"; stale.append(r["symbol"])
        else:
            r["status"] = "OK"
    req_bad = [s for s in stale + failed if s in required]
    status = "FAIL_CLOSED" if req_bad else "OK"
    stamp = started.strftime("%Y%m%dT%H%M%S")
    rep = {"run_at_et": started.isoformat(timespec="seconds"), "finished_at_et": now_et().isoformat(timespec="seconds"),
           "expected_last_complete_session": expected.isoformat(), "status": status,
           "required_bad": req_bad, "stale": stale, "dead": dead, "failed": failed,
           "n_symbols": len(rows), "code_version": CODE_VERSION, "rows": rows}
    jp = HEALTH / f"health_{stamp}.json"
    jp.write_text(json.dumps(rep, indent=2))
    lines = [f"# Data health — {started.isoformat(timespec='seconds')} ET", "",
             f"- Expected last completed session: **{expected}** (16:30 ET rule, NYSE holidays)",
             f"- Status: **{status}**" + (f" — required stale/failed: {', '.join(req_bad)}" if req_bad else ""),
             f"- Symbols: {len(rows)} | stale: {len(stale)} | dead/delisted: {len(dead)} | failed: {len(failed)}", "",
             "| Symbol | Status | Last bar | Lag (sessions) | Bars | Added | Action | Flags |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x.get("status") != "OK", x["symbol"])):
        iss = r.get("issues", {})
        flags = ", ".join(f"{k}={v}" for k, v in iss.items() if v) if iss else r.get("error", "")
        lines.append(f"| {r['symbol']} | {r.get('status')} | {r.get('last','')} | {r.get('lag_sessions','')} | {r.get('bars','')} | {r.get('added','')} | {r.get('action','')} | {flags} |")
    lines += ["", "abs_ret_gt_50pct counts single-day moves >50% (real for some meme/biotech names; check if unexpected).",
              "Fail-closed rule: any required (non-dead) symbol stale or failed -> exit code 2; downstream predictions must not run."]
    (HEALTH / "latest.md").write_text("\n".join(lines), encoding="utf-8")
    with (HEALTH / "refresh_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({k: rep[k] for k in ("run_at_et", "finished_at_et", "expected_last_complete_session", "status", "required_bad", "stale", "dead", "failed", "n_symbols")}) + "\n")
    return jp, (2 if req_bad else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=None)
    ap.add_argument("--universe", default=str(UNIVERSE))
    ap.add_argument("--overlap-days", type=int, default=14)
    ap.add_argument("--adj-tol", type=float, default=0.002)
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()
    started = now_et()
    expected = expected_last_complete_session(started)
    syms = [s.strip() for s in args.symbols.split(",")] if args.symbols else universe_symbols(Path(args.universe))
    rows = []
    for i, s in enumerate(syms):
        try:
            r = refresh_symbol(s, expected, args.overlap_days, args.adj_tol)
        except Exception as e:
            r = {"symbol": s, "ok": False, "error": repr(e)[:300]}
        rows.append(r)
        print(f"[{i+1}/{len(syms)}] {s}: {r.get('action')} last={r.get('last')} lag={r.get('lag_sessions')} added={r.get('added')} {r.get('error','')}", flush=True)
        time.sleep(args.sleep)
    # required = everything that is not dead
    required = {r["symbol"] for r in rows if not (r.get("ok") and r.get("lag_sessions", 0) > DEAD_AFTER_SESSIONS)}
    jp, code = write_health(rows, expected, started, required)
    print(f"HEALTH {jp} exit={code}")
    sys.exit(code)


if __name__ == "__main__":
    main()
