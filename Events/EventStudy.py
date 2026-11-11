"""Event-phase study (pre-registered in Events/Hypotheses.py). $10k, no deposits.
Parts: (A) underlying event study on actual adjusted prices with placebo sessions and date-cluster bootstrap;
(B) $10k account sims of the rule-based trades (shares; synthetic-option debit spreads/condors/straddles priced by
Options.Validate10k with real ticks/costs) - SYNTHETIC option prices, never real fills; (C) shares-momentum
event-avoidance overlays over all 10 weekday x parity phases; (D) H10 real 2026 option marks around events.
Signals use closes <= R-1 / R and fill at the next open; no same-bar exits.
Usage: .venv\\Scripts\\python -m Events.EventStudy --procs 8
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, time
from datetime import date, datetime
from multiprocessing import Pool
from pathlib import Path
import numpy as np

from Options import Validate10k as V
from Options import SynthBacktest as SB
from Options.Pricing import PayoffExtremes, Intrinsic
from Options.VolModel import IvScenario, ImpliedVol
from Options.RealHistory import ImpliedVolBisect
from Events import Hypotheses as HY

ROOT = SB.ROOT
BASE = "2026-11-09_event-options"
CODE_VERSION = "stonk-data-0.6.0-events"
MACRO_TK = ["SPY", "QQQ", "IWM", "TLT"]
EARN_TK = ["HOOD", "SHOP", "AXTI", "QURE"]
ACCOUNT = 10000.0
RNG = np.random.default_rng(11)
_D = {}


def LoadRegistry(dates):
    di = {d: i for i, d in enumerate(dates)}; ev = []
    import os
    with Path(os.environ.get("EVENT_REGISTRY", ROOT / "data/events/registry.csv")).open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["include"] != "True" or r["reaction_session"] not in di: continue
            r["R"] = di[r["reaction_session"]]; ev.append(r)
    return ev


def Window(tk, a_kind, a, b_kind, b):
    pa = tk[a_kind][a]; pb = tk[b_kind][b]
    return pb / pa - 1 if np.isfinite(pa) and np.isfinite(pb) and pa > 0 else np.nan


def Placebos(dates, excl_idx, s0, T, pad=3):
    bad = np.zeros(T, bool)
    for i in excl_idx: bad[max(0, i - pad):min(T, i + pad + 1)] = True
    return [t for t in range(max(s0, 7), T - 13) if not bad[t]]


def ClusterBoot(vals, clusters, n=1000):
    vals = np.asarray(vals, float); cl = np.asarray(clusters)
    ok = np.isfinite(vals); vals, cl = vals[ok], cl[ok]
    if len(vals) < 3: return (np.nan, np.nan, np.nan, len(vals), len(set(cl)))
    u = np.unique(cl); idx = {c: np.nonzero(cl == c)[0] for c in u}; ms = np.empty(n)
    for k in range(n):
        pick = RNG.choice(u, size=len(u), replace=True); ms[k] = np.concatenate([vals[idx[c]] for c in pick]).mean()
    return (float(vals.mean()), float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5)), len(vals), len(u))


def DiffBoot(a, ca, b, cb, n=1000):
    a, b = np.asarray(a, float), np.asarray(b, float); ca, cb = np.asarray(ca), np.asarray(cb)
    if len(a) < 3 or len(b) < 3: return (np.nan, np.nan, np.nan, np.nan)
    ua, ub = np.unique(ca), np.unique(cb); ia = {c: np.nonzero(ca == c)[0] for c in ua}; ib = {c: np.nonzero(cb == c)[0] for c in ub}
    ds = np.empty(n)
    for k in range(n):
        pa = RNG.choice(ua, len(ua)); pb = RNG.choice(ub, len(ub))
        ds[k] = np.concatenate([a[ia[c]] for c in pa]).mean() - np.concatenate([b[ib[c]] for c in pb]).mean()
    d = a.mean() - b.mean()
    return (float(d), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)), float(np.mean(ds <= 0)))  # one-sided p(diff<=0)


# ---------------- synthetic option trade ----------------
def OptTrade(sym, structure, t_in, t_out, exp, scen, cm, bump_in=0.0, ev_in=0.0, D=None, settle=False):
    """Enter at open t_in, exit at open t_out (< exp) or settle at exp close. Returns dict or None."""
    D = D or _D; dates, tk = D["dates"], D["tick"][sym]
    if not (np.isfinite(tk["Or"][t_in]) and np.isfinite(tk["base"][t_in - 1])) or tk["splitf"][t_in] != tk["splitf"][exp]: return None
    S = tk["Or"][t_in]; Ty = SB.Tyears(dates, t_in, exp, True); Se = S - V.PvDivs(tk, t_in, exp, None)
    sc_in = IvScenario(scen.name, scen.ivrp * (1 + bump_in), scen.skew)
    atm = ImpliedVol(tk["base"][t_in - 1], Se, Se, Ty, sc_in, ev_in, True, False)
    legs = V.Legs(sym, structure, S, atm, Ty); mn, mx = PayoffExtremes(legs)
    if not math.isfinite(mn): return None
    mids = V.Mids(sym, tk, legs, S, Se, tk["base"][t_in - 1], Ty, sc_in, ev_in, True)
    cost = V.Fill(sym, legs, mids, cm, True, S); ml = cost - mn * 100
    if ml <= 0: return None
    path = {}
    if settle: t_out = exp   # hold to expiry, cash-settle at intrinsic on the raw close (assignment/pin unmodeled)
    for t in range(t_in, min(t_out, exp) + 1):
        if t == t_out and not settle:
            S2 = tk["Or"][t] if np.isfinite(tk["Or"][t]) else tk["Cr"][t - 1]
            T2 = SB.Tyears(dates, t, exp, True); Se2 = S2 - V.PvDivs(tk, t, exp, None)
            m2 = V.Mids(sym, tk, legs, S2, Se2, tk["base"][t - 1], T2, scen, 0.0, False)
            proc = -V.Fill(sym, legs, m2, cm, False, S2); break
        S2 = tk["Cr"][t]; T2 = SB.Tyears(dates, t, exp, False)
        if t == exp:
            proc = sum(lg["qty"] * Intrinsic(S2, lg["K"], lg["kind"]) for lg in legs) * 100; break
        Se2 = S2 - V.PvDivs(tk, t, exp, None)
        m2 = V.Mids(sym, tk, legs, S2, Se2, tk["base"][t], T2, scen, 0.0, False)
        path[t] = 100 * SB.PackageMid(legs, m2, T2, mn, mx)
    return {"sym": sym, "t_in": t_in, "t_out": t, "cost": cost, "max_loss": ml, "proc": proc, "pnl": proc - cost, "ret_on_risk": (proc - cost) / ml,
            "path": path, "legs": ";".join(f"{lg['qty']:+d}{lg['kind']}{lg['K']:g}" for lg in legs)}


# ---------------- $10k account simulator over a list of trade specs ----------------
def Account(trades, kind, risk, D=None, cap_total=None):
    """trades: list of dicts with sym, t_in, t_out and either (shares) 'dir' or (options) cost/max_loss/proc/path.
    Shares: notional = risk*E/(2 sigma_d sqrt(h)), <=50% E per position, <=100% total, whole shares, 5bps*cm slip.
    Options: contracts = floor(min(risk*E, 0.5E, free, cap*E - open risk)/max_loss)."""
    D = D or _D; dates, tick = D["dates"], D["tick"]; T = len(dates); s0 = next(i for i, d in enumerate(dates) if d >= SB.SIM_START)
    by_in = {}
    for x in trades: by_in.setdefault(x["t_in"], []).append(x)
    cash = ACCOUNT; eq = np.full(T, np.nan); openp = []; done = []
    cap_total = cap_total or (1.0 if kind == "shares" else 0.25)
    for t in range(s0, T):
        for p in list(openp):
            if p["t_out"] == t:
                if kind == "shares":
                    px = tick[p["sym"]]["O"][t]; cash += p["n"] * px * (1 - p["slip"]); p["pnl"] = p["n"] * (px * (1 - p["slip"]) - p["entry"])
                else:
                    cash += p["n"] * p["proc"]; p["pnl"] = p["n"] * (p["proc"] - p["cost"])
                done.append(p); openp.remove(p)
        E = eq[t - 1] if np.isfinite(eq[t - 1]) else cash
        for x in by_in.get(t, []):
            if any(p["sym"] == x["sym"] for p in openp): continue
            if kind == "shares":
                tk = tick[x["sym"]]; px = tk["O"][t]; sd = tk["base"][t - 1] / math.sqrt(252); h = max(1, x["t_out"] - t)
                if not (np.isfinite(px) and np.isfinite(sd) and sd > 0): continue
                used = sum(p["n"] * tick[p["sym"]]["Cff"][t - 1] for p in openp)
                notional = min(risk * E / (2 * sd * math.sqrt(h)), 0.5 * E, cap_total * E - used, cash)
                slip = x["cm"] * SB.STOCK_SLIP_BPS / 1e4; n = int(notional // (px * (1 + slip)))
                if n < 1: continue
                cash -= n * px * (1 + slip); openp.append({**x, "n": n, "entry": px * (1 + slip), "slip": slip, "notional_frac": n * px / E})
            else:
                free = cash - sum(p["n"] * (p["max_loss"] - p["cost"]) for p in openp); orisk = sum(p["n"] * p["max_loss"] for p in openp)
                n = int(min(risk * E, 0.5 * E, free, cap_total * E - orisk) // x["max_loss"])
                if n < 1: continue
                cash -= n * x["cost"]; openp.append({**x, "n": n})
        mark = 0.0
        for p in openp:
            if kind == "shares": mark += p["n"] * tick[p["sym"]]["Cff"][t]
            else: mark += p["n"] * p["path"].get(t, p["cost"])
        cash += cash * SB.CASH_YIELD / 252 if cash > 0 else 0.0
        eq[t] = cash + mark
    return eq, done


def AcctMetrics(eq, done, D=None):
    D = D or _D; dates = D["dates"]; W = {k: SB.WinIdx(dates, *v) for k, v in SB.WINDOWS.items()}
    m = {w: SB.Metrics(eq, *W[w]) for w in ("ALL", "PRE", "HO")}
    s0 = W["ALL"][0]; pn = np.array([p["pnl"] for p in done]) if done else np.array([])
    w, l = pn[pn > 0], pn[pn <= 0]
    return {"n_trades": len(done), "final_equity": float(eq[-1]), "ALL_cagr": m["ALL"]["cagr"], "ALL_sharpe": m["ALL"]["sharpe"], "ALL_maxdd": m["ALL"]["maxdd"],
            "PRE_cagr": m["PRE"]["cagr"], "PRE_sharpe": m["PRE"]["sharpe"], "HO_cagr": m["HO"]["cagr"], "HO_sharpe": m["HO"]["sharpe"],
            "min_equity": float(np.nanmin(eq[s0:])), "severe_lt50": bool(np.nanmin(eq[s0:]) < 0.5 * ACCOUNT), "ruin_lt10": bool(np.nanmin(eq[s0:]) < 0.1 * ACCOUNT),
            "win_rate": float(np.mean(pn > 0)) if len(pn) else np.nan, "payoff_ratio": float(w.mean() / -l.mean()) if len(w) and len(l) and l.mean() < 0 else np.nan}


# ---------------- shares momentum with event overlays ----------------
def MomentumOverlay(cm, weekday, parity, flat_days=None, reenter_days=None, earn_R=None, D=None):
    D = D or _D; dates, tick = D["dates"], D["tick"]; Tn = len(dates); s0 = next(i for i, d in enumerate(dates) if d >= SB.SIM_START)
    wo = [(date.fromisoformat(d) - date(2000, 1, 3)).days // 7 for d in dates]; wd = [date.fromisoformat(d).weekday() for d in dates]
    reb, seen = [], set()
    for t in range(1, Tn):
        if wo[t] % 2 == parity and wd[t] >= weekday and wo[t] not in seen: reb.append(t); seen.add(wo[t])
    nxt = {reb[i]: (reb[i + 1] if i + 1 < len(reb) else Tn) for i in range(len(reb))}; rebs = set(reb)
    flat_days = flat_days or set(); reenter_days = reenter_days or set(); earn_R = earn_R or {}
    cash = ACCOUNT; qty = {}; target_w = {}; eq = np.full(Tn, np.nan); flat = False; n_skip = 0
    def trade_to(t, wts):
        nonlocal cash, qty
        E = cash + sum(q * tick[s]["Cff"][t - 1] for s, q in qty.items())
        tgt = {s: w * E / tick[s]["O"][t] for s, w in wts.items() if np.isfinite(tick[s]["O"][t])}
        for s in set(qty) | set(tgt):
            o = tick[s]["O"][t] if np.isfinite(tick[s]["O"][t]) else tick[s]["Cff"][t - 1]
            dq = tgt.get(s, 0.0) - qty.get(s, 0.0); cash -= dq * o + abs(dq) * o * cm * SB.STOCK_SLIP_BPS / 1e4; qty[s] = tgt.get(s, 0.0)
        qty = {s: q for s, q in qty.items() if q > 0}
    for t in range(s0, Tn):
        if t in rebs:
            picks = [s for s in SB.TICKERS if s in tick and tick[s]["bull"][t - 1] and np.isfinite(tick[s]["O"][t])]
            if earn_R:
                keep = [s for s in picks if not any(t <= r < nxt[t] for r in earn_R.get(s, []))]; n_skip += len(picks) - len(keep); picks = keep
            w = min(0.5, 1.0 / len(picks)) if picks else 0.0; target_w = {s: w for s in picks}
        if t in flat_days and not flat: trade_to(t, {}); flat = True
        elif t in reenter_days and flat: trade_to(t, target_w); flat = False
        elif t in rebs and not flat: trade_to(t, target_w)
        cash += cash * SB.CASH_YIELD / 252 if cash > 0 else 0.0
        eq[t] = cash + sum(q * tick[s]["Cff"][t] for s, q in qty.items())
    return eq, n_skip


def _mom_job(a):
    cm, wd, par, mode = a; ev = _D["events"]
    if mode == "base": return a, MomentumOverlay(cm, wd, par)[0], 0
    if mode == "H7_macro_avoid":
        Rs = [e["R"] for e in ev if e["event_type"] in ("FOMC", "CPI")]
        return a, MomentumOverlay(cm, wd, par, flat_days={r - 1 for r in Rs}, reenter_days={r + 1 for r in Rs})[0], 0
    er = {}
    for e in ev:
        if e["event_type"] == "EARNINGS": er.setdefault(e["symbol"], []).append(e["R"])
    eq, ns = MomentumOverlay(cm, wd, par, earn_R=er); return a, eq, ns


def _init():
    import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
    V._init(); _D.update(V._D); _D["events"] = LoadRegistry(_D["dates"])


def _w(path, rows, cols=None):
    if not rows: path.write_text("no rows\n"); return
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in rows: w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


def RealIvEvents(ev, dates):
    """H10: change in real inverted IV close(R-1)->close(R) around events vs non-event consecutive pairs (2026 panel)."""
    panel = json.loads((ROOT / "data/calibration/rh_option_daily_20261109_v2.json").read_text())
    raw = {}; rows = []
    macroR = {dates[e["R"]] for e in ev if e["event_type"] in ("FOMC", "CPI")}; macro_type = {dates[e["R"]]: e["event_type"] for e in ev if e["event_type"] in ("FOMC", "CPI")}
    earnR = {}
    for e in ev:
        if e["event_type"] == "EARNINGS": earnR.setdefault(e["symbol"], set()).add(dates[e["R"]])
    di = {d: i for i, d in enumerate(dates)}
    for c in panel["contracts"]:
        s = c["symbol"]
        if s not in raw: raw[s] = {b["date"]: b["close"] for b in json.loads((ROOT / "data/daily_raw" / f"{s}.json").read_text())["bars"]}
        bars = {b["date"]: b["close"] for b in c["bars"] if not b["interpolated"] and b["date"] <= c["expiry"]}
        ds = sorted(bars); exp = date.fromisoformat(c["expiry"])
        def iv(d):
            S = raw[s].get(d); days = (exp - date.fromisoformat(d)).days
            if S is None or days < 4: return None
            intr = max(S - c["strike"], 0) if c["type"] == "C" else max(c["strike"] - S, 0)
            if bars[d] - intr < 0.02: return None
            return ImpliedVolBisect(bars[d], S, c["strike"], days / 365, c["type"])
        evset = earnR.get(s, set()) if s in EARN_TK else macroR
        near = set()
        for d in evset:
            if d in di:
                for k in range(-2, 3):
                    j = di[d] + k
                    if 0 <= j < len(dates): near.add(dates[j])
        for d in ds:
            if d not in di or di[d] == 0: continue
            dp = dates[di[d] - 1]
            if dp not in bars: continue
            a, b = iv(dp), iv(d)
            if a is None or b is None: continue
            kind = ("EVENT_" + (macro_type.get(d, "EARNINGS") if s not in EARN_TK else "EARNINGS")) if d in evset else ("placebo" if d not in near else None)
            if kind is None: continue
            rows.append({"symbol": s, "occ_symbol": c["occ_symbol"], "date": d, "pair_kind": kind, "iv_prev": a, "iv": b, "d_iv_pts": (b - a) * 100, "rel_change": b / a - 1,
                         "basis": "RH daily option bar close (not bid/ask), unadjusted spot"})
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=8); ap.add_argument("--fill", default="v2"); ap.add_argument("--tag", default="")
    a = ap.parse_args(); V.FILL_MODEL = a.fill
    t0 = time.time(); run_at = datetime.now(SB.ET); run_id = f"ev_{run_at.strftime('%Y%m%dT%H%M%S')}{a.tag}"
    rd = SB.OUT / run_id; rd.mkdir(parents=True, exist_ok=False)
    hyp_sha = hashlib.sha256((ROOT / "Events/Hypotheses.py").read_bytes()).hexdigest()
    _init(); D = _D; dates = D["dates"]; T = len(dates); ev = D["events"]; tick = D["tick"]
    s0 = next(i for i, d in enumerate(dates) if d >= SB.SIM_START)
    _w(rd / f"{BASE}_hypothesis-registry.csv", [{**h, "tickers": json.dumps(h["tickers"]), "status": "tested"} for h in HY.HYPOTHESES] +
       [{"id": x["id"], "rule": "EXCLUDED", "status": "excluded", "placebo": "", "kind": "", "events": "", "tickers": "", "why": x["why"]} for x in HY.EXCLUDED])
    # ---- H10 first (feeds the H8 bump) ----
    h10 = RealIvEvents(ev, dates); _w(rd / f"{BASE}_h10-real-iv-pairs.csv", h10)
    h10s = []
    for grp in ("ETF", "SINGLE"):
        for k in sorted({r["pair_kind"] for r in h10}):
            sub = [r for r in h10 if r["pair_kind"] == k and ((r["symbol"] in EARN_TK) == (grp == "SINGLE"))]
            if not sub: continue
            h10s.append({"group": grp, "pair_kind": k, "n_pairs": len(sub), "n_dates": len({r["date"] for r in sub}), "median_d_iv_pts": float(np.median([r["d_iv_pts"] for r in sub])),
                         "median_rel_change": float(np.median([r["rel_change"] for r in sub])), "mean_rel_change": float(np.mean([r["rel_change"] for r in sub]))})
    _w(rd / f"{BASE}_h10-real-iv-summary.csv", h10s)
    def med(grp, kinds):
        v = [r["rel_change"] for r in h10 if r["pair_kind"] in kinds and ((r["symbol"] in EARN_TK) == (grp == "SINGLE"))]; return float(np.median(v)) if v else np.nan
    pl = med("ETF", ("placebo",)); evm = med("ETF", ("EVENT_FOMC", "EVENT_CPI"))
    bump = max(0.0, -(evm - pl)) if np.isfinite(evm) and np.isfinite(pl) else 0.0   # observed excess crush only; never forced
    # ---- (A) underlying event study ----
    macro_R = sorted({e["R"] for e in ev if e["event_type"] in ("FOMC", "CPI")})
    earn_R = {s: sorted({e["R"] for e in ev if e["event_type"] == "EARNINGS" and e["symbol"] == s}) for s in EARN_TK}
    plac_macro = Placebos(dates, macro_R, s0, T)
    est, trades_spec = [], {}
    def study_rows(kind, tks, Rs_by_tk, hold, sig_mult, plac_by_tk):
        out = []
        for s in tks:
            tk = tick[s]
            for R, eid in Rs_by_tk[s]:
                if R - 6 < 0 or R + 1 + hold >= T or R < s0: continue
                sd = tk["base"][R - 1] / math.sqrt(252); react = Window(tk, "C", R - 1, "C", R)
                out.append({"kind": kind, "event_id": eid, "symbol": s, "R_date": dates[R], "R": R, "sigma_d_tminus1": sd, "pre5": Window(tk, "C", R - 6, "C", R - 1),
                            "h1_window_open_Rm1_to_open_Rp1": Window(tk, "O", R - 1, "O", R + 1), "reaction": react, "react_z": react / sd if sd > 0 else np.nan,
                            "post_open_Rp1_h": Window(tk, "O", R + 1, "O", R + 1 + hold), "hold": hold, "sig_mult": sig_mult,
                            "window": "HO" if dates[R] >= "2025-01-31" else ("EARLY" if dates[R] <= "2017-12-31" else "IS")})
        return out
    mac = {e["R"]: e for e in ev if e["event_type"] in ("FOMC", "CPI")}
    for typ, hid in (("FOMC", "H2"), ("CPI", "H3")):
        Rs = {s: [(e["R"], e["event_id"]) for e in ev if e["event_type"] == typ] for s in MACRO_TK}
        est += study_rows(typ, MACRO_TK, Rs, HY.HOLD_MACRO, HY.SIGMA_MACRO, None)
    Rs = {s: [(e["R"], e["event_id"]) for e in ev if e["event_type"] == "EARNINGS" and e["symbol"] == s] for s in EARN_TK}
    est += study_rows("EARNINGS", EARN_TK, Rs, HY.HOLD_EARN, HY.SIGMA_EARN, None)
    # placebo rows
    pla = []
    for s in MACRO_TK:
        pla += study_rows("PLACEBO_MACRO", [s], {s: [(t, f"PL_{s}_{dates[t]}") for t in plac_macro]}, HY.HOLD_MACRO, HY.SIGMA_MACRO, None)
    for s in EARN_TK:
        pe = Placebos(dates, earn_R[s] + macro_R, s0, T)
        pe = [t for t in pe if np.isfinite(tick[s]["C"][t - 6])]
        pla += study_rows("PLACEBO_EARN", [s], {s: [(t, f"PL_{s}_{dates[t]}") for t in pe]}, HY.HOLD_EARN, HY.SIGMA_EARN, None)
    _w(rd / f"{BASE}_event-study.csv", [{k: v for k, v in r.items() if k != "R"} for r in est])
    # hypothesis-level tests (gross & net of 2x round-trip slip at cm=2)
    res = []
    def add(hid, sub_ev, sub_pl, fn, label, cm=2.0):
        rt = 2 * cm * SB.STOCK_SLIP_BPS / 1e4
        ev_v = [fn(r) for r in sub_ev]; pl_v = [fn(r) for r in sub_pl]
        ev_n = [(v - rt if v == v and v != 0 else v) for v in ev_v]; pl_n = [(v - rt if v == v and v != 0 else v) for v in pl_v]
        keep_e = [(v, r["R_date"]) for v, r in zip(ev_n, sub_ev) if v == v and v != 0]; keep_p = [(v, r["R_date"]) for v, r in zip(pl_n, sub_pl) if v == v and v != 0]
        if not keep_e: res.append({"hypothesis": hid, "config": label, "n_trades": 0}); return
        m, lo, hi, n, nc = ClusterBoot([x[0] for x in keep_e], [x[1] for x in keep_e])
        d, dlo, dhi, p = DiffBoot([x[0] for x in keep_e], [x[1] for x in keep_e], [x[0] for x in keep_p], [x[1] for x in keep_p]) if keep_p else (np.nan,) * 4
        byw = {}
        for v, r in zip(ev_n, sub_ev):
            if v == v and v != 0: byw.setdefault(r["window"], []).append(v)
        res.append({"hypothesis": hid, "config": label, "n_trades": n, "n_date_clusters": nc, "mean_net": m, "ci95_lo": lo, "ci95_hi": hi,
                    "hit_rate": float(np.mean([x[0] > 0 for x in keep_e])), "placebo_n": len(keep_p), "placebo_mean_net": float(np.mean([x[0] for x in keep_p])) if keep_p else np.nan,
                    "diff_vs_placebo": d, "diff_ci_lo": dlo, "diff_ci_hi": dhi, "p_one_sided": p,
                    **{f"mean_{w}": float(np.mean(v)) for w, v in byw.items()}, **{f"n_{w}": len(v) for w, v in byw.items()}})
    fomc = [r for r in est if r["kind"] == "FOMC"]; cpi = [r for r in est if r["kind"] == "CPI"]; ern = [r for r in est if r["kind"] == "EARNINGS"]
    plm = [r for r in pla if r["kind"] == "PLACEBO_MACRO"]; ple = [r for r in pla if r["kind"] == "PLACEBO_EARN"]
    add("H1_fomc_preevent_long", [r for r in fomc if r["symbol"] == "SPY"], [r for r in plm if r["symbol"] == "SPY"], lambda r: r["h1_window_open_Rm1_to_open_Rp1"], "SPY long open(R-1)->open(R+1)")
    for hid, sub in (("H2_fomc_reaction_continuation", fomc), ("H3_cpi_reaction_continuation", cpi)):
        thr = lambda r: abs(r["react_z"]) > HY.SIGMA_MACRO
        add(hid, [r for r in sub if thr(r)], [r for r in plm if thr(r)], lambda r: math.copysign(1, r["reaction"]) * r["post_open_Rp1_h"], "signed (long/short study, not cash-acct tradable)")
        add(hid, [r for r in sub if thr(r) and r["reaction"] > 0], [r for r in plm if thr(r) and r["reaction"] > 0], lambda r: r["post_open_Rp1_h"], "long-only after up-reaction (tradable)")
        add(hid, [r for r in sub if r["reaction"] > 0], [r for r in plm if r["reaction"] > 0], lambda r: r["post_open_Rp1_h"], "NO-THRESHOLD long after any up day (sensitivity)")
    thr = lambda r: abs(r["react_z"]) > HY.SIGMA_EARN
    add("H4_earnings_reaction_continuation", [r for r in ern if thr(r)], [r for r in ple if thr(r)], lambda r: math.copysign(1, r["reaction"]) * r["post_open_Rp1_h"], "signed")
    add("H4_earnings_reaction_continuation", [r for r in ern if thr(r) and r["reaction"] > 0], [r for r in ple if thr(r) and r["reaction"] > 0], lambda r: r["post_open_Rp1_h"], "long-only after up-reaction")
    add("H5_earnings_reaction_reversal", [r for r in ern if thr(r)], [r for r in ple if thr(r)], lambda r: -math.copysign(1, r["reaction"]) * r["post_open_Rp1_h"], "signed")
    add("H5_earnings_reaction_reversal", [r for r in ern if thr(r) and r["reaction"] < 0], [r for r in ple if thr(r) and r["reaction"] < 0], lambda r: r["post_open_Rp1_h"], "long-only after down-reaction")
    # pre-event drift descriptive (proxy, NOT an expectation)
    for k, sub, ps in (("FOMC", fomc, plm), ("CPI", cpi, plm), ("EARNINGS", ern, ple)):
        add(f"DESCRIPTIVE_pre5_drift_{k}", sub, ps, lambda r: r["pre5"] + 2 * 2 * SB.STOCK_SLIP_BPS / 1e4, "trailing 5d drift before event (proxy; gross)")
        add(f"DESCRIPTIVE_abs_reaction_{k}", sub, ps, lambda r: abs(r["reaction"]) + 2 * 2 * SB.STOCK_SLIP_BPS / 1e4, "|reaction| event vs placebo (gross)")
    # ---- (B) $10k account sims ----
    acct_rows, eq_series, trade_rows = [], {}, []
    def specs_shares(sub, hold, sign_fn):
        out = []
        for r in sub:
            if sign_fn(r) > 0: out.append({"sym": r["symbol"], "t_in": r["R"] + 1, "t_out": r["R"] + 1 + hold, "event_id": r["event_id"]})
        return out
    groups = {
        "H1_fomc_preevent_long": ([{"sym": "SPY", "t_in": r["R"] - 1, "t_out": r["R"] + 1, "event_id": r["event_id"]} for r in fomc if r["symbol"] == "SPY"], None),
        "H2_fomc_reaction_continuation": (specs_shares([r for r in fomc if abs(r["react_z"]) > HY.SIGMA_MACRO], HY.HOLD_MACRO, lambda r: r["reaction"]), [r for r in fomc if abs(r["react_z"]) > HY.SIGMA_MACRO]),
        "H3_cpi_reaction_continuation": (specs_shares([r for r in cpi if abs(r["react_z"]) > HY.SIGMA_MACRO], HY.HOLD_MACRO, lambda r: r["reaction"]), [r for r in cpi if abs(r["react_z"]) > HY.SIGMA_MACRO]),
        "H4_earnings_reaction_continuation": (specs_shares([r for r in ern if abs(r["react_z"]) > HY.SIGMA_EARN], HY.HOLD_EARN, lambda r: r["reaction"]), [r for r in ern if abs(r["react_z"]) > HY.SIGMA_EARN]),
        "H5_earnings_reaction_reversal": (specs_shares([r for r in ern if abs(r["react_z"]) > HY.SIGMA_EARN], HY.HOLD_EARN, lambda r: -r["reaction"]), [r for r in ern if abs(r["react_z"]) > HY.SIGMA_EARN]),
    }
    for hid, (sp, opt_src) in groups.items():
        for cm in HY.COSTS:
            for rk in HY.SIZING:
                eq, done = Account([{**x, "cm": cm} for x in sp], "shares", rk)
                acct_rows.append({"hypothesis": hid, "instrument": "shares (actual prices)", "cost_mult": cm, "risk": rk, "ivrp": "", **AcctMetrics(eq, done)})
                if cm == 2.0 and rk == 0.02:
                    eq_series[f"{hid}|shares|r02|cost2x"] = eq
                    trade_rows += [{"hypothesis": hid, "instrument": "shares", "symbol": p["sym"], "entry_date": dates[p["t_in"]], "exit_date": dates[p["t_out"]], "qty": p["n"], "pnl": p["pnl"], "event_id": p["event_id"]} for p in done]
        if opt_src is None: continue
        hold = HY.HOLD_MACRO if hid[:2] in ("H2", "H3") else HY.HOLD_EARN
        for sc_m, settle in [(m_, st_) for m_ in HY.IVRPS for st_ in (False, True)]:
            scen = IvScenario(f"iv{sc_m:.2f}_sk10", sc_m, 0.10)
            for cm in HY.COSTS:
                ot = []
                for r in opt_src:
                    sgn = (1 if r["reaction"] > 0 else -1) * (-1 if hid.startswith("H5") else 1)
                    st = "bull_call_debit" if sgn > 0 else "bear_put_debit"; t_in = r["R"] + 1; exp = min(T - 1, t_in + 10); t_out = min(t_in + hold, exp)
                    x = OptTrade(r["symbol"], st, t_in, t_out if t_out < exp else exp, exp, scen, cm, settle=settle)
                    if x: ot.append({**x, "event_id": r["event_id"], "structure": st})
                for rk in HY.SIZING:
                    eq, done = Account(ot, "options", rk)
                    rr = [x["ret_on_risk"] for x in ot]
                    acct_rows.append({"hypothesis": hid, "instrument": "SYNTHETIC debit spreads (model prices)", "exit_mode": "settle_at_expiry(post-hoc fix)" if settle else "open_exit(prereg; ITM half-spread artifact)", "cost_mult": cm, "risk": rk, "ivrp": sc_m,
                                      "per_trade_mean_ret_on_risk": float(np.mean(rr)) if rr else np.nan, **AcctMetrics(eq, done)})
                    if cm == 2.0 and rk == 0.02 and sc_m == 1.1:
                        eq_series[f"{hid}|SYNTH_debit|{'settle' if settle else 'openexit'}|r02|cost2x|iv1.10"] = eq
                        trade_rows += [{"hypothesis": hid, "instrument": "SYNTHETIC_option", "exit_mode": "settle" if settle else "open_exit", "symbol": p["sym"], "entry_date": dates[p["t_in"]], "exit_date": dates[p["t_out"]],
                                        "qty": p["n"], "pnl": p["pnl"], "event_id": p["event_id"], "legs": p["legs"], "structure": p["structure"]} for p in done]
    # H8 SPY condor across macro events (synthetic) + placebo; H9 earnings straddle research comparator
    plac_spy = plac_macro[::5]
    for sc_m, settle in [(m_, st_) for m_ in HY.IVRPS for st_ in (False, True)]:
        scen = IvScenario(f"iv{sc_m:.2f}_sk10", sc_m, 0.10); xm = "settle_R+4(post-hoc)" if settle else "open_R+1(prereg)"
        for cm in HY.COSTS:
            for b in sorted({0.0, round(bump, 4)}):
                ot = []
                for R in macro_R:
                    if R - 1 < s0 or R + 4 >= T: continue
                    x = OptTrade("SPY", "iron_condor", R - 1, R + 1, R + 4, scen, cm, bump_in=b, settle=settle)
                    if x: ot.append({**x, "event_id": mac[R]["event_id"], "structure": "iron_condor"})
                pt = [OptTrade("SPY", "iron_condor", t - 1, t + 1, t + 4, scen, cm, settle=settle) for t in plac_spy if t + 4 < T]
                pt = [x for x in pt if x]
                d, dlo, dhi, p = DiffBoot([x["ret_on_risk"] for x in ot], [dates[x["t_in"]] for x in ot], [x["ret_on_risk"] for x in pt], [dates[x["t_in"]] for x in pt])
                for rk in HY.SIZING:
                    eq, done = Account(ot, "options", rk)
                    acct_rows.append({"hypothesis": "H8_spy_event_condor_SYNTHETIC", "instrument": f"SYNTHETIC iron condor, event IV bump {b}", "exit_mode": xm, "cost_mult": cm, "risk": rk, "ivrp": sc_m,
                                      "per_trade_mean_ret_on_risk": float(np.mean([x["ret_on_risk"] for x in ot])), "placebo_mean_ret_on_risk": float(np.mean([x["ret_on_risk"] for x in pt])) if pt else np.nan,
                                      "diff_vs_placebo": d, "diff_ci_lo": dlo, "diff_ci_hi": dhi, **AcctMetrics(eq, done)})
                    if cm == 2.0 and rk == 0.02 and sc_m == 1.1 and b == max({0.0, round(bump, 4)}):
                        eq_series[f"H8|SYNTH_condor|{xm}|r02|cost2x|iv1.10|bump{b}"] = eq
            ot = []
            for s in EARN_TK:
                for R in earn_R[s]:
                    if R - 1 < s0 or R + 4 >= T: continue
                    tk = tick[s]; past = [abs(math.log(tk["C"][r] / tk["C"][r - 1])) for r in earn_R[s] if r < R - 1 and np.isfinite(tk["C"][r]) and np.isfinite(tk["C"][r - 1])]
                    evv = (np.mean(past) if past else SB.DEFAULT_JUMP) ** 2
                    x = OptTrade(s, "long_straddle", R - 1, R + 1, R + 4, scen, cm, ev_in=evv, settle=settle)
                    if x: ot.append({**x, "event_id": f"EARN_{s}_{dates[R]}", "structure": "long_straddle"})
            for rk in HY.SIZING:
                eq, done = Account(ot, "options", rk)
                acct_rows.append({"hypothesis": "H9_earnings_straddle_RESEARCH_COMPARATOR", "instrument": "SYNTHETIC long straddle across earnings (RESEARCH ONLY)", "exit_mode": xm, "cost_mult": cm, "risk": rk, "ivrp": sc_m,
                                  "per_trade_mean_ret_on_risk": float(np.mean([x["ret_on_risk"] for x in ot])) if ot else np.nan, **AcctMetrics(eq, done)})
    # ---- (C) momentum overlays, all phases ----
    jobs = [(cm, wd, par, mode) for cm in HY.COSTS for wd in range(5) for par in (0, 1) for mode in ("base", "H6_earn_avoid", "H7_macro_avoid")]
    with Pool(a.procs, initializer=_init) as pool: mres = pool.map(_mom_job, jobs)
    W = {k: SB.WinIdx(dates, *v) for k, v in SB.WINDOWS.items()}; mrows = []; meq = {}
    for (cm, wd, par, mode), eq, _ in mres:
        m = {w: SB.Metrics(eq, *W[w]) for w in ("ALL", "PRE", "HO")}; meq[(cm, wd, par, mode)] = eq
        mrows.append({"mode": mode, "cost_mult": cm, "weekday": wd, "parity": par, "ALL_cagr": m["ALL"]["cagr"], "ALL_sharpe": m["ALL"]["sharpe"], "ALL_maxdd": m["ALL"]["maxdd"],
                      "PRE_sharpe": m["PRE"]["sharpe"], "HO_sharpe": m["HO"]["sharpe"], "min_equity": float(np.nanmin(eq[s0:]))})
    for r in mrows:
        b = next(x for x in mrows if x["mode"] == "base" and x["cost_mult"] == r["cost_mult"] and x["weekday"] == r["weekday"] and x["parity"] == r["parity"])
        r["d_ALL_cagr_vs_base"] = r["ALL_cagr"] - b["ALL_cagr"]; r["d_PRE_sharpe_vs_base"] = r["PRE_sharpe"] - b["PRE_sharpe"]; r["d_ALL_maxdd_vs_base"] = r["ALL_maxdd"] - b["ALL_maxdd"]
    _w(rd / f"{BASE}_momentum-overlays.csv", mrows)
    for mode in ("base", "H6_earn_avoid", "H7_macro_avoid"): eq_series[f"shares_momentum|{mode}|wd0p0|cost2x"] = meq[(2.0, 0, 0, mode)]
    spy = tick["SPY"]["Cff"].astype(float); spy_eq = np.full(T, np.nan); spy_eq[s0 - 1:] = spy[s0 - 1:] / spy[s0 - 1] * ACCOUNT
    eq_series["SPY_buyhold_totalreturn"] = spy_eq; eq_series["cash_2pct"] = SB.RunCash(ACCOUNT, D)["eq"]
    for nm in ("SPY_buyhold_totalreturn", "cash_2pct"):
        acct_rows.append({"hypothesis": "BENCHMARK", "instrument": nm, **AcctMetrics(eq_series[nm], [])})
    # Holm over the primary tradable configs
    prim = [r for r in res if r.get("config", "").startswith(("SPY long", "long-only")) and r.get("p_one_sided") == r.get("p_one_sided") and r.get("n_trades", 0) > 0]
    prim.sort(key=lambda r: r["p_one_sided"]); mtests = len(prim)
    for i, r in enumerate(prim): r["holm_threshold_0.05"] = 0.05 / (mtests - i); r["holm_pass"] = r["p_one_sided"] <= 0.05 / (mtests - i) and all(x["p_one_sided"] <= 0.05 / (mtests - j) for j, x in enumerate(prim[:i]))
    _w(rd / f"{BASE}_hypothesis-results.csv", res); _w(rd / f"{BASE}_account-results.csv", acct_rows); _w(rd / f"{BASE}_trades.csv", trade_rows)
    eq_rows = []
    for k, e in eq_series.items():
        e = np.asarray(e, float); pk = np.fmax.accumulate(np.where(np.isfinite(e), e, -np.inf))
        for t in range(s0 - 1, T):
            if np.isfinite(e[t]): eq_rows.append({"date": dates[t], "series": k, "equity": round(float(e[t]), 2), "drawdown_pct": round(float(e[t] / pk[t] - 1), 5)})
    _w(rd / f"{BASE}_equity.csv", eq_rows)
    from collections import Counter
    meta = {"run_id": run_id, "run_at_et": run_at.isoformat(timespec="seconds"), "code_version": CODE_VERSION, "hypotheses_sha256": hyp_sha, "end": SB.END,
            "n_events_included": dict(Counter(e["event_type"] for e in ev)), "n_placebo_macro_sessions": len(plac_macro), "h8_event_iv_bump_from_H10": bump,
            "h10_medians": {"etf_event": evm, "etf_placebo": pl}, "account": ACCOUNT, "deposits": 0, "holm_m": mtests, "elapsed_s": time.time() - t0,
            "notes": "synthetic option results are model prices (Validate10k engine, real ticks, 1/2/3x costs); real 2026 marks only in H10"}
    (rd / f"{BASE}_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps(meta, indent=1, default=str)); print("DONE", rd)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
    import Events.EventStudy as _M; _M.main()
