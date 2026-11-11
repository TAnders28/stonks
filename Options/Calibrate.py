"""Contemporary calibration check (NOT historical validation): compare model IV/prices/Greeks against
read-only Robinhood quotes saved in data/calibration/rh_calib_<date>.json.

Usage: .venv\\Scripts\\python -m Options.Calibrate --file data/calibration/rh_calib_20261109.json --out <csv>
"""
from __future__ import annotations
import argparse, csv, json, math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np

from Options.Pricing import BsPrice, BsGreeks
from Options.VolModel import TrailingVol, IvScenario, ImpliedVol
from Options import SynthBacktest as SB

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(ROOT / "data/calibration/rh_calib_20261109.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    js = json.loads(Path(a.file).read_text())
    qt = datetime.fromisoformat(js["spot_time_utc"].replace("Z", "+00:00")).astimezone(ET)
    rows = []; base_cache = {}
    for r in js["rows"]:
        s = r["symbol"]
        if s not in base_cache:
            bars = json.loads((ROOT / "data/daily" / f"{s}.json").read_text())["bars"]
            c = np.array([b["close"] for b in bars]); v = TrailingVol(c)
            base_cache[s] = (float(v["base"][-1]), float(v["rv20"][-1]), bars[-1]["date"])
        base, rv20, asof = base_cache[s]
        S = js["spot"][s]
        exp = datetime.fromisoformat(r["expiry"] + "T16:00:00").replace(tzinfo=ET)
        T = (exp - qt).total_seconds() / (365 * 86400)
        k = "C" if r["type"] == "call" else "P"
        mid = (r["bid"] + r["ask"]) / 2
        out = {"symbol": s, "type": r["type"], "strike": r["strike"], "expiry": r["expiry"], "spot": S, "T_days": round(T * 365, 2),
               "bid": r["bid"], "ask": r["ask"], "mid": mid, "market_iv": r["market_iv"], "market_delta": r["delta"],
               "half_spread_pct_market": (r["ask"] - r["bid"]) / 2 / mid if mid > 0 else None,
               "half_spread_pct_assumed_1x": SB.HALF_SPREAD_PCT.get(s), "trailing_base_vol": base, "rv20": rv20, "vol_asof_close": asof,
               "market_iv_over_base": (r["market_iv"] / base) if r["market_iv"] else None}
        for m in (1.0, 1.2, 1.4):
            iv = ImpliedVol(base, S, r["strike"], T, IvScenario("c", m, 0.10))
            out[f"model_iv_{m}"] = iv
            out[f"model_mid_{m}"] = BsPrice(S, r["strike"], T, SB.RATE, 0.0, iv, k)
        g = BsGreeks(S, r["strike"], T, SB.RATE, 0.0, r["market_iv"] or out["model_iv_1.2"], k)
        out["bs_delta_at_market_iv"] = g["delta"]; out["bs_mid_at_market_iv"] = g["price"]
        rows.append(out)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    for s in sorted({r["symbol"] for r in rows}):
        rr = [r for r in rows if r["symbol"] == s]
        atm = min(rr, key=lambda r: abs(r["strike"] - r["spot"]))
        print(f"{s}: ATM K={atm['strike']} market_iv={atm['market_iv']:.3f} trailing_base={atm['trailing_base_vol']:.3f} ratio={atm['market_iv_over_base']:.2f} "
              f"mid={atm['mid']:.2f} model1.2={atm['model_mid_1.2']:.2f} BS@mktIV={atm['bs_mid_at_market_iv']:.2f} "
              f"halfspread mkt={atm['half_spread_pct_market']:.3f} assumed={atm['half_spread_pct_assumed_1x']}")
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
