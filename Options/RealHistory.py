"""Real-history check of the synthetic IV model on a SPARSE Robinhood option panel (read-only data).

- Observed: Robinhood daily option price bars (marks/trades; NOT executable bid/ask; no volume). Interpolated
  (gap-fill) bars are dropped; bars after expiry dropped.
- Spot: UNADJUSTED raw close (data/daily_raw) matching listed strikes. Forecast vol: trailing base vol from the
  adjusted series using closes up to t-1 only (strictly past).
- IV inversion: BS bisection (r=2%, q=0) at the 16:00 close; invalid when price outside no-arb bounds,
  extrinsic < 2 ticks, or < 1 calendar day to expiry.
- Held-back test: the best fixed IVRP multiplier is fit on observations dated <= SPLIT_DATE and evaluated on later
  dates only. A 46-contract panel over Mar-Nov 2026 cannot validate a 2012-2026 surface.
Usage: .venv\\Scripts\\python -m Options.RealHistory --out-dir <dir> --base <basename>
"""
from __future__ import annotations
import argparse, csv, json, math
from datetime import date
from pathlib import Path
import numpy as np

from Options.Pricing import BsPrice, PriceBounds
from Options.VolModel import TrailingVol

ROOT = Path(__file__).resolve().parents[1]
RATE = 0.02
SPLIT_DATE = "2026-07-31"
MULTS = [round(0.8 + 0.05 * k, 2) for k in range(15)]  # 0.80..1.50


def Tick(sym, px, ticks):
    lo, hi = ticks or (0.01, 0.05)
    return lo if px < 3 else hi


def ImpliedVolBisect(price, S, K, T, kind, r=RATE):
    lo_b, hi_b = PriceBounds(S, K, T, r, 0.0, kind)
    if not (lo_b < price < hi_b):
        return None
    a, b = 1e-3, 6.0
    for _ in range(80):
        m = 0.5 * (a + b)
        if BsPrice(S, K, T, r, 0.0, m, kind) > price: b = m
        else: a = m
    return 0.5 * (a + b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(ROOT / "data/calibration/rh_option_daily_20261109.json"))
    ap.add_argument("--out-dir", required=True); ap.add_argument("--base", required=True)
    a = ap.parse_args(); od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    panel = json.loads(Path(a.file).read_text())
    raw, adjv = {}, {}
    rows, cov = [], []
    for c in panel["contracts"]:
        s = c["symbol"]
        if s not in raw:
            rj = json.loads((ROOT / "data/daily_raw" / f"{s}.json").read_text())
            raw[s] = {b["date"]: b["close"] for b in rj["bars"]}
            aj = json.loads((ROOT / "data/daily" / f"{s}.json").read_text())["bars"]
            v = TrailingVol(np.array([b["close"] for b in aj]))
            dates = [b["date"] for b in aj]
            adjv[s] = {dates[i]: (float(v["base"][i - 1]) if i > 0 else float("nan")) for i in range(len(dates))}  # forecast known at t-1 close
        kind = c["type"]; K = c["strike"]; exp = date.fromisoformat(c["expiry"])
        n_real = n_valid = 0; reasons = {"interpolated": 0, "after_expiry": 0, "no_spot": 0, "lt_1day": 0, "bounds": 0, "low_extrinsic": 0}
        for b in c["bars"]:
            if b["interpolated"]: reasons["interpolated"] += 1; continue
            if b["date"] > c["expiry"]: reasons["after_expiry"] += 1; continue
            n_real += 1
            S = raw[s].get(b["date"]); f = adjv[s].get(b["date"])
            if S is None or f is None or not math.isfinite(f): reasons["no_spot"] += 1; continue
            days = (exp - date.fromisoformat(b["date"])).days
            if days < 1: reasons["lt_1day"] += 1; continue
            T = days / 365.0; px = b["close"]
            intrinsic = max(S - K, 0) if kind == "C" else max(K - S, 0)
            tick = Tick(s, px, c.get("min_ticks_below3_above3"))
            if px - intrinsic < 2 * tick: reasons["low_extrinsic"] += 1; continue
            iv = ImpliedVolBisect(px, S, K, T, kind)
            if iv is None: reasons["bounds"] += 1; continue
            n_valid += 1
            row = {"date": b["date"], "symbol": s, "occ_symbol": c["occ_symbol"], "type": kind, "strike": K, "expiry": c["expiry"],
                   "days_to_exp": days, "spot_raw": S, "moneyness": K / S, "obs_price": px, "obs_iv": iv, "forecast_base_vol_tminus1": f,
                   "iv_over_forecast": iv / f, "tick": tick, "tick_pct_of_price": tick / px, "half": "fit" if b["date"] <= SPLIT_DATE else "test"}
            for m in (1.0, 1.2, 1.4):
                mp = BsPrice(S, K, T, RATE, 0.0, f * m, kind); row[f"model_px_m{m}"] = mp; row[f"err_pct_m{m}"] = mp / px - 1
            rows.append(row)
        cov.append({"occ_symbol": c["occ_symbol"], "symbol": s, "expiry": c["expiry"], "type": kind, "strike": K, "n_bars_returned": c["n_bars"],
                    "n_interpolated": c["n_interpolated"], "n_real_bars_to_expiry": n_real, "n_valid_iv": n_valid, **{f"drop_{k}": v for k, v in reasons.items()},
                    "basis": "obs=RH option bar close (not bid/ask); spot=unadjusted close", "source": panel["source"], "fetched_at_et": panel["fetched_at_et"]})
    def W(name, rr):
        with (od / f"{a.base}_{name}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rr[0].keys())); w.writeheader()
            for r in rr: w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    W("realhist-coverage", cov); W("realhist-iv", rows)
    # summary + held-back fit
    summ = []
    for s in sorted({r["symbol"] for r in rows}) + ["ALL"]:
        rr = [r for r in rows if s == "ALL" or r["symbol"] == s]
        fit = [r for r in rr if r["half"] == "fit"]; test = [r for r in rr if r["half"] == "test"]
        def mape(sub, m):
            out = []
            for r in sub:
                S, K, T = r["spot_raw"], r["strike"], r["days_to_exp"] / 365
                out.append(abs(BsPrice(S, K, T, RATE, 0, r["forecast_base_vol_tminus1"] * m, r["type"]) / r["obs_price"] - 1))
            return float(np.median(out)) if out else float("nan")
        best = min(MULTS, key=lambda m: mape(fit, m)) if fit else float("nan")
        ratio = np.array([r["iv_over_forecast"] for r in rr])
        summ.append({"symbol": s, "n_obs": len(rr), "n_fit": len(fit), "n_test": len(test), "median_iv_over_forecast": float(np.median(ratio)),
                     "p25": float(np.percentile(ratio, 25)), "p75": float(np.percentile(ratio, 75)),
                     "median_ratio_fit": float(np.median([r["iv_over_forecast"] for r in fit])) if fit else float("nan"),
                     "median_ratio_test": float(np.median([r["iv_over_forecast"] for r in test])) if test else float("nan"),
                     "best_mult_on_fit": best, "test_medAPE_at_best": mape(test, best) if test and fit else float("nan"),
                     "test_medAPE_m1.0": mape(test, 1.0), "test_medAPE_m1.2": mape(test, 1.2), "test_medAPE_m1.4": mape(test, 1.4),
                     "median_tick_pct_of_price": float(np.median([r["tick_pct_of_price"] for r in rr]))})
    W("realhist-summary", summ)
    for r in summ:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
