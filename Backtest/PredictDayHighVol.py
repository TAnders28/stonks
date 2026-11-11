"""Forward signals for selected high-vol strategies (read-only)."""
from __future__ import annotations
import argparse, json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from Backtest.Strategies.HighVolFamilies import (
    build_variant_grid, sma,
)

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data" / "daily"
OUT = ROOT / "out"
UNIVERSE = ROOT / "data/universe/highvol_2026-11-08.json"


def load_bars(sym: str):
    p = DAILY / f"{sym.replace('^','')}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["bars"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--winners-json", default="", help="path to results.json from hunt; uses winners[].signal+regime")
    ap.add_argument("--max-bar-age-days", type=int, default=4)
    args = ap.parse_args()

    u = json.loads(UNIVERSE.read_text())
    exclude = {"leveraged_inverse", "benchmarks", "vol_proxy"}
    trade_buckets = {}
    for bname, b in u["buckets"].items():
        if bname in exclude:
            continue
        syms = [s for s in b["symbols"] if load_bars(s) and len(load_bars(s)) >= 100]
        if syms:
            trade_buckets[bname] = syms
    variants = {s.name: s for _, s in build_variant_grid(trade_buckets, max_w=0.50)}

    # default winners if no results yet
    selected = [
        {"signal": "xs_mom_lb63_top5_skip1", "regime": "spy200"},
        {"signal": "rsi2_mr_lt10_trend100_top5", "regime": "spy200"},
    ]
    if args.winners_json:
        res = json.loads(Path(args.winners_json).read_text())
        selected = [{"signal": w["signal"], "regime": w["regime"], "name": w["name"]} for w in res.get("winners", [])]

    spy = load_bars("SPY")
    vix = load_bars("VIX")
    now = datetime.now(ET)
    last_spy = spy[-1]["date"]
    age = (now.date() - datetime.fromisoformat(last_spy).date()).days
    stale = age > args.max_bar_age_days

    spy_closes = [b["close"] for b in spy]
    spy200 = sma(spy_closes, 200)
    above = spy200 is not None and spy_closes[-1] > spy200
    vix_last = vix[-1]["close"] if vix else None

    histories = {}
    i_by_sym = {}
    for syms in trade_buckets.values():
        for s in syms:
            bars = load_bars(s)
            if not bars:
                continue
            histories[s] = bars
            i_by_sym[s] = len(bars) - 1

    out_picks = []
    for sel in selected:
        sig = variants.get(sel["signal"])
        if not sig:
            out_picks.append({"error": f"unknown signal {sel['signal']}", **sel})
            continue
        regime = sel["regime"]
        if regime == "off":
            regime_ok = True
        elif regime == "spy200":
            regime_ok = bool(above)
        else:
            regime_ok = bool(above) and (vix_last is not None and vix_last < 25)
        if stale:
            weights = {}
            note = "NO_SIGNAL_STALE"
        else:
            weights = sig(histories, i_by_sym, regime_ok)
            note = "OK"
        out_picks.append({
            "strategy": sel.get("name", f"{sel['signal']}__reg_{sel['regime']}"),
            "signal": sel["signal"],
            "regime": regime,
            "regime_ok": regime_ok,
            "spy_last": spy_closes[-1],
            "spy_above_200": above,
            "vix_last": vix_last,
            "as_of_bar_date": last_spy,
            "bar_age_days": age,
            "stale": stale,
            "status": note,
            "targets": weights,
            "execution_note": "If trading equities: next session open. Options: treat as path-bias research only under freeze.",
        })

    stamp = now.strftime("%Y%m%dT%H%M%S")
    out_dir = OUT / f"predict_highvol_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    payload = {"run_at_et": now.isoformat(timespec="seconds"), "picks": out_picks}
    (out_dir / "picks.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    if stale:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
