"""Fast numpy daily engine for cross-sectional momentum on the high-vol universe.

Semantics (leak-free by construction):
- Signal computed at CLOSE of day t from data <= t; orders fill at OPEN of t+1.
- Momentum = C[t-skip]/C[t-skip-lb]-1 (skip=1). Only names with positive momentum are eligible.
- Top-N equal weight, weight = min(0.50, 1/len(picks)) (Tim's 50%-per-position cap); remainder cash (0% yield).
- Rebalance calendar-anchored: W = last session of each week, 2W = last session of even ISO weeks,
  M = last session of month. Session calendar is the SPY calendar (known in advance).
- Trailing stop: held name whose close < peak_close_since_entry*(1-stop) is sold next open; it is excluded
  from picks until the next rebalance (may be re-picked then).
- Overlays (risk control): spy200 = go to cash at next open when SPY close < SMA200, re-enter (fresh ranking)
  the day after it recovers; vt40 = at each rebalance scale gross exposure to min(1, 40%/ex-ante vol) using a
  60-day covariance of the picks.
- Earnings skip (optional): at a rebalance, a name is ineligible if its earnings release would be absorbed
  before the next rebalance fill (BMO day d -> session d; AMC day d -> next session). Unknown (pre-coverage) = not skipped.
- Legacy flags (diagnostics only): peak_reset, fallen_excl reproduce the quirks of the mt_ engine.
- Costs: per-bucket bps each way on traded notional (from the universe file). Fractional shares.
"""
from __future__ import annotations
import json, math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data" / "daily"
EARN = ROOT / "data" / "earnings"
UNIVERSE = ROOT / "data" / "universe" / "highvol_2026-11-08.json"
EXCLUDE_BUCKETS = {"leveraged_inverse", "benchmarks", "vol_proxy"}
CODE_VERSION = "stonk-data-0.3.0-vecmom"


def load_bars(sym):
    p = DAILY / f"{sym.replace('^', '')}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["bars"]


def universe(path=UNIVERSE):
    u = json.loads(Path(path).read_text())
    syms, bps, bucket = [], {}, {}
    for bname, b in u["buckets"].items():
        if bname in EXCLUDE_BUCKETS:
            continue
        for s in b["symbols"]:
            if s not in bucket:
                syms.append(s); bucket[s] = bname; bps[s] = float(b.get("default_cost_bps", 12))
    return syms, bps, bucket


@dataclass
class Panel:
    dates: list
    syms: list
    O: np.ndarray
    C: np.ndarray
    Cff: np.ndarray
    spy_c: np.ndarray
    bps: np.ndarray
    bucket: list


def load_panel(start="2007-01-01", end=None, min_bars=100):
    spy = load_bars("SPY")
    spy = [b for b in spy if b["date"] >= start and (end is None or b["date"] <= end)]
    dates = [b["date"] for b in spy]
    di = {d: i for i, d in enumerate(dates)}
    syms_all, bps_map, bucket_map = universe()
    syms, Os, Cs = [], [], []
    for s in syms_all:
        bars = load_bars(s)
        if not bars or len(bars) < min_bars:
            continue
        o = np.full(len(dates), np.nan); c = np.full(len(dates), np.nan)
        for b in bars:
            i = di.get(b["date"])
            if i is not None and b["open"] > 0 and b["close"] > 0:
                o[i] = b["open"]; c[i] = b["close"]
        if np.isfinite(c).sum() < min_bars:
            continue
        syms.append(s); Os.append(o); Cs.append(c)
    O = np.column_stack(Os); C = np.column_stack(Cs)
    Cff = pd.DataFrame(C).ffill().values
    spy_c = np.array([b["close"] for b in spy])
    return Panel(dates, syms, O, C, Cff, spy_c, np.array([bps_map[s] for s in syms]), [bucket_map[s] for s in syms])


def shift(a, k):
    out = np.full_like(a, np.nan)
    if k < len(a):
        out[k:] = a[:-k] if k > 0 else a
    return out


def reb_masks(dates):
    d = [date.fromisoformat(x) for x in dates]
    T = len(d)
    from DataPull.market_calendar import next_session
    nxt = d[1:] + [next_session(d[-1])]
    wk = [x.isocalendar()[:2] for x in d]; nwk = [x.isocalendar()[:2] for x in nxt]
    W = np.array([wk[i] != nwk[i] for i in range(T)])
    W2 = np.array([W[i] and (wk[i][1] % 2 == 0) for i in range(T)])
    M = np.array([d[i].month != nxt[i].month for i in range(T)])
    return {"W": W, "2W": W2, "M": M}


def build_features(P: Panel, lbs=(63, 126, 189, 252)):
    C = P.C
    feats = {"mom": {lb: shift(C, 1) / shift(C, 1 + lb) - 1 for lb in lbs}}
    lr = np.log(C / shift(C, 1))
    feats["lr"] = np.nan_to_num(lr, nan=0.0)
    sma200 = pd.Series(P.spy_c).rolling(200).mean().values
    feats["spy_ok"] = np.where(np.isfinite(sma200), P.spy_c >= sma200, True)
    feats["reb"] = reb_masks(P.dates)
    return feats


def load_earnings_block(P: Panel, rebmask: np.ndarray):
    """bool T x N: True = earnings would land inside the holding window starting at fill t+1 until next reb fill."""
    T, N = P.C.shape
    di = {d: i for i, d in enumerate(P.dates)}
    from DataPull.market_calendar import next_session
    blk = np.zeros((T, N), dtype=bool)
    covered = np.zeros((T, N), dtype=bool)
    reb_idx = np.where(rebmask)[0]
    # next reb fill index for each t
    nxt_fill = np.full(T, T - 1)
    for t in range(T):
        k = np.searchsorted(reb_idx, t, side="right")
        nxt_fill[t] = min(T - 1, reb_idx[k] + 1) if k < len(reb_idx) else T - 1
    for n, s in enumerate(P.syms):
        p = EARN / f"{s}.json"
        if not p.exists():
            continue
        rec = json.loads(p.read_text())
        if not rec.get("dates"):
            continue
        first = rec["coverage_first"]
        eff = []
        for r in rec["dates"]:
            d = date.fromisoformat(r["date"])
            e = d if r["timing"] == "BMO" else next_session(d)
            if e.isoformat() in di:
                eff.append(di[e.isoformat()])
        eff = np.array(sorted(set(eff)))
        f0 = np.searchsorted(np.array(P.dates), first)
        covered[f0:, n] = True
        if len(eff) == 0:
            continue
        for t in range(T - 1):
            lo, hi = t + 1, nxt_fill[t]
            k = np.searchsorted(eff, lo, side="right")
            if k < len(eff) and eff[k] <= hi:
                blk[t, n] = True
    return blk, covered


def simulate(P: Panel, feats, cfg, start_idx, end_idx, earn_block=None, record=False, equity0=10_000.0):
    T, N = P.C.shape
    mom = feats["mom"][cfg["lb"]]
    reb = feats["reb"][cfg["reb"]]
    ov = cfg.get("ov", "none")
    use_spy = "spy200" in ov
    use_vt = "vt40" in ov
    vt_target = 0.40
    spy_ok = feats["spy_ok"]; lr = feats["lr"]
    stop = cfg.get("stop", 0.0) or 0.0
    top = cfg["top"]
    bps = P.bps / 1e4
    cash = equity0; qty = np.zeros(N); peak = np.zeros(N)
    entry_px = np.zeros(N); entry_dt = [None] * N
    eq = np.full(T, np.nan); expo = np.zeros(T); entries = np.zeros(T, dtype=np.int32); turn = np.zeros(T)
    stopped_until_reb = np.zeros(N, dtype=bool)
    peak_reset = bool(cfg.get("peak_reset"))      # legacy mt_ engine: reset all peaks to open at every execution
    fallen_excl = bool(cfg.get("fallen_excl"))    # legacy mt_ engine: never re-buy a former holding while close < its last peak*(1-stop)
    last_peak = np.full(N, np.nan)
    pending = None; prev_risk = None; trades = []; last_targets = np.zeros(N)
    for t in range(start_idx, end_idx + 1):
        o = P.O[t]; c = P.C[t]
        if pending is not None:
            kind, arg = pending
            tradable = np.isfinite(o)
            mark_o = np.where(tradable, o, P.Cff[t - 1]); mark_o = np.nan_to_num(mark_o)
            E = cash + float(qty @ mark_o)
            if kind == "w":
                safe_o = np.where(tradable, o, 1.0)
                tq = np.where(tradable & (arg > 0), arg * E * 0.998 / safe_o, 0.0)
                tq = np.where(tradable, tq, qty)
            else:
                tq = np.where(arg & tradable, 0.0, qty)
            dq = tq - qty
            # skip tiny resizes of names kept (< 2% of equity) to limit churn
            keep = (qty > 0) & (tq > 0) & (np.abs(dq) * mark_o < 0.02 * E)
            dq = np.where(keep, 0.0, dq); tq = qty + dq
            notional = np.abs(dq) * mark_o
            cost = float(notional @ bps)
            cash -= float(dq @ mark_o) + cost
            turn[t] = notional.sum() / max(E, 1e-9)
            new_in = (qty <= 0) & (tq > 0)
            entries[t] = int(new_in.sum())
            if record:
                for n in np.where(dq != 0)[0]:
                    px = mark_o[n]
                    if dq[n] > 0:
                        buy_px = px * (1 + bps[n])
                        if qty[n] <= 0:
                            entry_px[n] = buy_px; entry_dt[n] = P.dates[t]
                        else:
                            entry_px[n] = (qty[n] * entry_px[n] + dq[n] * buy_px) / tq[n]
                    else:
                        sell_px = px * (1 - bps[n]); sq = -dq[n]
                        trades.append({"entry_date": entry_dt[n], "exit_date": P.dates[t], "symbol": P.syms[n], "side": "long",
                                       "qty": sq, "entry_price": entry_px[n], "exit_price": sell_px,
                                       "pnl": sq * (sell_px - entry_px[n])})
            peak = np.where(new_in, mark_o, peak)
            if peak_reset:
                peak = np.where(tq > 1e-12, mark_o, peak)
            qty = np.where(tq > 1e-12, tq, 0.0)
            pending = None
        mark = np.nan_to_num(np.where(np.isfinite(c), c, P.Cff[t]))
        hv = float(qty @ mark)
        E_c = cash + hv
        eq[t] = E_c; expo[t] = hv / E_c if E_c > 0 else 0.0
        held = qty > 0
        peak = np.where(held & np.isfinite(c), np.maximum(peak, np.nan_to_num(c)), peak)
        if fallen_excl:
            last_peak = np.where(held, peak, last_peak)
        risk = bool(spy_ok[t]) if use_spy else True
        stopped = (held & np.isfinite(c) & (np.nan_to_num(c) < peak * (1 - stop))) if stop > 0 else np.zeros(N, dtype=bool)
        if reb[t] or (risk and prev_risk is False):
            stopped_until_reb[:] = False
            if risk:
                m = mom[t]
                elig = np.isfinite(m) & (m > 0) & np.isfinite(c) & ~stopped
                if fallen_excl and stop > 0:
                    elig &= ~(np.isfinite(last_peak) & (np.nan_to_num(c) < last_peak * (1 - stop)))
                if earn_block is not None:
                    elig &= ~earn_block[t]
                idx = np.where(elig)[0]
                order = idx[np.argsort(-m[idx], kind="stable")][:top]
                w = np.zeros(N)
                if len(order):
                    w[order] = min(0.5, 1.0 / len(order))
                    if use_vt and t >= 60:
                        R = lr[t - 59:t + 1][:, order]
                        cov = np.atleast_2d(np.cov(R, rowvar=False)) * 252
                        wo = w[order]
                        pv = math.sqrt(max(float(wo @ cov @ wo), 1e-12))
                        w *= min(1.0, vt_target / pv)
                pending = ("w", w); last_targets = w
            else:
                pending = ("w", np.zeros(N)); last_targets = np.zeros(N)
        elif not risk and held.any():
            pending = ("w", np.zeros(N)); last_targets = np.zeros(N)
        elif stopped.any():
            stopped_until_reb |= stopped
            pending = ("drop", stopped)
        prev_risk = risk
    res = {"eq": eq, "expo": expo, "entries": entries, "turn": turn}
    if record:
        res["trades"] = trades
        res["final_pending"] = pending
        res["final_qty"] = qty; res["final_cash"] = cash
        res["last_targets"] = last_targets
    return res


def window_metrics(eq, expo, entries, s, e):
    """metrics on [s, e] using eq[s-1] as base if available."""
    base = eq[s - 1] if s > 0 and np.isfinite(eq[s - 1]) else eq[s]
    seg = np.concatenate([[base], eq[s:e + 1]])
    r = seg[1:] / seg[:-1] - 1
    n = len(r)
    tot = seg[-1] / seg[0]
    cagr = tot ** (252 / n) - 1 if tot > 0 else -1.0
    sd = r.std(ddof=1)
    sh = r.mean() / sd * math.sqrt(252) if sd > 0 else float("nan")
    pk = np.maximum.accumulate(seg); dd = (seg / pk - 1).min()
    return {"cagr": float(cagr), "sharpe": float(sh), "maxdd": float(dd), "vol": float(sd * math.sqrt(252)),
            "expo": float(expo[s:e + 1].mean()), "entries": int(entries[s:e + 1].sum()), "n_days": n}
