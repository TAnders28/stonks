"""Abstract edge search (pre-registered in Abstract/2026-11-09_abstract-edge-search_hypotheses.json).
Usage: .venv\\Scripts\\python -m Abstract.Search --procs 6
Writes out/ae_<stamp>/ : comparison.csv, equity.csv, trades.csv, f1_*.csv, meta.json.
"""
from __future__ import annotations
import argparse, csv, json, hashlib, math, time
from datetime import date, datetime
from multiprocessing import Pool
from pathlib import Path
import numpy as np

from Options import SynthBacktest as SB
from Options import Validate10k as V
from Options.VolModel import IvScenario

ROOT = Path(__file__).resolve().parents[1]
HYP = ROOT / "Abstract/2026-11-09_abstract-edge-search_hypotheses.json"
ETFS = ["SPY", "QQQ", "IWM", "XBI", "GDX", "TLT"]
CASH = SB.CASH_YIELD; ACC = 10000.0
CEN = IvScenario("iv1.10_sk10", 1.1, 0.10)
VIXS = IvScenario("vix9d_sk10", 1.0, 0.10)
_D = {}


def init(fm, itm):
    V.FILL_MODEL, V.ITM_FRAC = fm, itm
    if not _D:
        import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
        _D.update(V.LoadData2()); V._D.update(_D)
        dates = _D["dates"]; di = {d: i for i, d in enumerate(dates)}
        vix = np.full(len(dates), np.nan)
        for b in json.loads((ROOT / "data/vol_index/VIX9D.json").read_text())["bars"]:
            if b["date"] in di: vix[di[b["date"]]] = b["close"] / 100.0
        for i in range(1, len(vix)):
            if not np.isfinite(vix[i]): vix[i] = vix[i - 1]
        _D["vix9d"] = vix
        C = _D["tick"]["SPY"]["Cff"]; lr = np.r_[np.nan, np.diff(np.log(C))]
        rv = np.full(len(C), np.nan)
        for t in range(21, len(C)): rv[t] = np.std(lr[t - 19:t + 1], ddof=1) * math.sqrt(252)
        _D["ratio"] = vix / rv


def S0(dates): return next(i for i, d in enumerate(dates) if d >= SB.SIM_START)


# ---------------- options jobs ----------------
def opt_job(a):
    key, struct, sizing, cm, tenor, scen_kind, gate, thr, itm = a
    V.ITM_FRAC = itm
    D = _D
    if scen_kind == "vix9d":
        tk = dict(D["tick"]["SPY"]); tk["base"] = D["vix9d"]
        ent = D["entries_by_tenor"][tenor]
        r = D["ratio"]
        if gate == "short": ent = {t: e for t, e in ent.items() if np.isfinite(r[t - 1]) and r[t - 1] >= thr}
        elif gate == "long": ent = {t: e for t, e in ent.items() if np.isfinite(r[t - 1]) and r[t - 1] <= thr}
        D2 = {"dates": D["dates"], "tick": {"SPY": tk}, "entries_by_tenor": {tenor: ent}}
        res = V.RunOptions2(V.Cfg2(struct, "expiry", sizing, VIXS, cm, tenor, record=True), D2)
    else:
        sc = {"c": CEN, "iv100": IvScenario("iv1.00_sk10", 1.0, 0.10), "iv125": IvScenario("iv1.25_sk10", 1.25, 0.10)}[scen_kind]
        res = V.RunOptions2(V.Cfg2(struct, "expiry", sizing, sc, cm, tenor, record=True), D)
    tr = res.pop("trades", [])
    return key, res, tr


# ---------------- shares engines ----------------
def run_weights(weight_fn, reb, cm, syms, D):
    """Generic long-only target-weight engine on adjusted prices. weight_fn(t) -> {sym: w} using data <= t-1."""
    dates, tick = D["dates"], D["tick"]; Tn = len(dates); s0 = S0(dates)
    cash = ACC; qty = {}; eq = np.full(Tn, np.nan); turn = 0.0; expo = np.zeros(Tn); ntr = 0; trades = []; min_cash = ACC
    for t in range(s0, Tn):
        if t in reb:
            px = {s: (tick[s]["O"][t] if np.isfinite(tick[s]["O"][t]) else tick[s]["Cff"][t - 1]) for s in set(qty) | set(syms) if s in tick}
            E = (cash + sum(q * px[s] for s, q in qty.items())) / (1 + cm * SB.STOCK_SLIP_BPS / 1e4 * 2)   # value at today's open, net of worst-case round-trip slippage
            w = weight_fn(t)
            assert all(v <= 0.5 + 1e-9 for s, v in w.items() if s != "SPY_BENCH") and sum(w.values()) <= 1.0 + 1e-9 or "SPY_BENCH" in w
            w = {("SPY" if s == "SPY_BENCH" else s): v for s, v in w.items()}
            tgt = {s: v * E / tick[s]["O"][t] for s, v in w.items() if v > 0 and np.isfinite(tick[s]["O"][t])}
            for s in set(qty) | set(tgt):
                o = tick[s]["O"][t] if np.isfinite(tick[s]["O"][t]) else tick[s]["Cff"][t - 1]
                dq = tgt.get(s, 0.0) - qty.get(s, 0.0)
                if abs(dq * o) < 1e-6: continue
                cash -= dq * o + abs(dq) * o * cm * SB.STOCK_SLIP_BPS / 1e4; turn += abs(dq) * o; ntr += 1
                trades.append({"date": dates[t], "symbol": s, "shares": round(dq, 4), "price": round(o, 4), "notional": round(dq * o, 2)})
                qty[s] = tgt.get(s, 0.0)
            qty = {s: q for s, q in qty.items() if q > 1e-12}
        if cash > 0: cash += cash * CASH / 252
        inv = sum(q * tick[s]["Cff"][t] for s, q in qty.items()); eq[t] = cash + inv; expo[t] = inv / eq[t]
        assert cash > -1.0, f"negative cash (leverage) {dates[t]} {cash}"   # <$1 float/rounding residue tolerated (reported as min_cash)
        min_cash = min(min_cash, cash)
    return {"eq": eq, "avg_exposure": float(expo[s0:].mean()), "turnover_per_yr": float(turn / ((Tn - s0) / 252) / np.nanmean(eq[s0:])), "n_fills": ntr, "trades": trades, "min_cash": min_cash}


def month_reb(dates, offset):
    out, seen = set(), set()
    for t in range(1, len(dates)):
        d = date.fromisoformat(dates[t]); k = (d.year, d.month)
        if d.day >= offset and k not in seen: out.add(t); seen.add(k)
    return out


def f2_weights(D, L, target):
    tick = D["tick"]
    lr = {s: np.r_[np.nan, np.diff(np.log(tick[s]["Cff"]))] for s in ETFS}
    def fn(t):
        el = []
        for s in ETFS:
            C = tick[s]["Cff"]
            if t - 1 - L < 0 or not np.isfinite(C[t - 1 - L]) or not np.isfinite(C[t - 1 - 21]): continue
            if C[t - 1 - 21] / C[t - 1 - L] - 1 > 0: el.append(s)
        if not el: return {}
        R = np.array([lr[s][t - 60:t] for s in el])
        if not np.all(np.isfinite(R)): return {}
        vol = R.std(axis=1, ddof=1) * math.sqrt(252); w = (1 / vol) / (1 / vol).sum()
        pv = math.sqrt(max(w @ np.atleast_2d(np.cov(R)) @ w * 252, 1e-12)); w = w * min(target / pv, 1e9)
        w = np.minimum(w, 0.5)
        if w.sum() > 1: w = w / w.sum()
        return {s: float(x) for s, x in zip(el, w)}
    return fn


def fixed_weights(w): return lambda t: dict(w)


def biweekly_reb(dates, weekday, parity):
    wo = [(date.fromisoformat(d) - date(2000, 1, 3)).days // 7 for d in dates]; wd = [date.fromisoformat(d).weekday() for d in dates]
    reb, done = set(), set()
    for t in range(1, len(dates)):
        if wo[t] % 2 == parity and wd[t] >= weekday and wo[t] not in done: reb.add(t); done.add(wo[t])
    return reb


def bear_weights(D):
    tick = D["tick"]
    def fn(t):
        p = [s for s in SB.TICKERS if s in tick and tick[s]["bear"][t - 1] and np.isfinite(tick[s]["O"][t])]
        return {s: min(0.5, 1.0 / len(p)) for s in p} if p else {}
    return fn


def tom_sched(dates, a, b):
    """entry at open of (last session of month m) + 1 + a ; exit at open of session (first session of m+1) + b"""
    first = [t for t in range(1, len(dates)) if dates[t][:7] != dates[t - 1][:7]]
    ins, outs = set(), set()
    for f in first:
        e, x = f + a, f + b
        if e >= 1 and x < len(dates): ins.add(e); outs.add(x)
    return ins, outs


def tom_weights(ins, outs, syms):
    state = {"on": False}
    def fn(t):
        if t in ins: state["on"] = True
        if t in outs: state["on"] = False
        return {s: 1 / len(syms) for s in syms} if state["on"] else {}
    return fn


# ---------------- metrics ----------------
def rets(eq): r = np.full(len(eq), np.nan); r[1:] = eq[1:] / eq[:-1] - 1; return r


def block_boot(x, reps=2000, block=21, seed=7):
    x = x[np.isfinite(x)]; n = len(x); rng = np.random.default_rng(seed); nb = int(math.ceil(n / block)); m = np.empty(reps)
    for i in range(reps):
        st = rng.integers(0, n, nb); idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:n] % n; m[i] = x[idx].mean()
    return m


def metrics(r, spy_r, dates, lo, hi):
    idx = [i for i, d in enumerate(dates) if lo <= d <= hi and np.isfinite(r[i])]
    x = r[idx]; s = spy_r[idx]; c = (1 + CASH) ** (1 / 252) - 1
    eqf = ACC * np.prod(1 + x); yrs = len(x) / 252; cagr = (eqf / ACC) ** (1 / yrs) - 1
    path = np.cumprod(1 + x); mdd = float(np.min(path / np.maximum.accumulate(path) - 1))
    vol = x.std(ddof=1) * math.sqrt(252); sh = (x.mean() - c) / x.std(ddof=1) * math.sqrt(252) if x.std() > 0 else float("nan")
    k = vol / (s.std(ddof=1) * math.sqrt(252)); vm = k * s + (1 - k) * c
    vm_cagr = np.prod(1 + vm) ** (1 / yrs) - 1
    ex = x - c; diff = x - vm
    be = block_boot(ex); bd = block_boot(diff)
    return {"start": dates[idx[0]], "end": dates[idx[-1]], "final_equity": round(eqf, 2), "cagr": cagr, "vol": vol, "sharpe": sh, "max_dd": mdd,
            "vm_spy_cagr": vm_cagr, "vm_k": k, "excess_vs_vmspy_ann": diff.mean() * 252,
            "p_excess_cash": float(np.mean(be - ex.mean() >= ex.mean())), "p_vs_vmspy": float(np.mean(bd - diff.mean() >= diff.mean())),
            "ci95_vs_vmspy_ann_lo": float(np.percentile(bd, 2.5) * 252), "ci95_vs_vmspy_ann_hi": float(np.percentile(bd, 97.5) * 252)}


def start_sens(r, dates, years):
    out = {}
    for y in years:
        idx = [i for i, d in enumerate(dates) if d >= f"{y}-01-01" and np.isfinite(r[i])]
        if len(idx) < 252: continue
        x = r[idx]; out[y] = float(np.prod(1 + x) ** (252 / len(x)) - 1)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=6); a = ap.parse_args()
    t0 = time.time(); run_at = datetime.now(SB.ET); rid = f"ae_{run_at.strftime('%Y%m%dT%H%M%S')}"
    out = ROOT / "out" / rid; out.mkdir(parents=True)
    hyp_sha = hashlib.sha256(HYP.read_bytes()).hexdigest()
    init("v3", 0.10); D = _D; dates = D["dates"]; Tn = len(dates)
    elig = [s for s in V.STRUCTS if V.InverseAllowed(s)]
    direc = [s for s in V.STRUCTS if V.Bias(s) in ("bull", "bear")]
    jobs = []
    for s in V.STRUCTS:
        for cm in (0.0, 1.0, 2.0, 3.0):
            jobs.append((("orig", s, cm, "2w_even", "c", 0.10), s, "r02c25", cm, "2w_even", "c", None, None, 0.10))
    for s in elig:
        for cm in (0.0, 1.0, 2.0, 3.0):
            jobs.append((("inv", s, cm, "2w_even", "c", 0.10), "inv:" + s, "r02c25", cm, "2w_even", "c", None, None, 0.10))
        jobs.append((("inv", s, 2.0, "2w_odd", "c", 0.10), "inv:" + s, "r02c25", 2.0, "2w_odd", "c", None, None, 0.10))
        jobs.append((("inv", s, 2.0, "2w_even", "c", 0.25), "inv:" + s, "r02c25", 2.0, "2w_even", "c", None, None, 0.25))
        for sk in ("iv100", "iv125"):
            jobs.append((("inv", s, 2.0, "2w_even", sk, 0.10), "inv:" + s, "r02c25", 2.0, "2w_even", sk, None, None, 0.10))
            jobs.append((("orig", s, 2.0, "2w_even", sk, 0.10), s, "r02c25", 2.0, "2w_even", sk, None, None, 0.10))
    for s in direc:
        for cm in (1.0, 2.0, 3.0):
            jobs.append((("opp", s, cm, "2w_even", "c", 0.10), "opp:" + s, "r02c25", cm, "2w_even", "c", None, None, 0.10))
        jobs.append((("opp", s, 2.0, "2w_odd", "c", 0.10), "opp:" + s, "r02c25", 2.0, "2w_odd", "c", None, None, 0.10))
    F3 = {"F3a": ("iron_condor", "short", 1.25), "F3b": ("bull_put_credit", "short", 1.25), "F3c": ("long_straddle", "long", 0.90),
          "F3d": ("iron_condor", None, None), "F3e": ("long_straddle", None, None)}
    NB3 = {"short": [1.1, 1.4], "long": [0.8, 1.0]}
    for k, (s, g, th) in F3.items():
        for cm in (1.0, 2.0, 3.0):
            jobs.append(((k, s, cm, "2w_even", "vix9d", 0.10, "r02c25", th), s, "r02c25", cm, "2w_even", "vix9d", g, th, 0.10))
        jobs.append(((k, s, 2.0, "2w_odd", "vix9d", 0.10, "r02c25", th), s, "r02c25", 2.0, "2w_odd", "vix9d", g, th, 0.10))
        jobs.append(((k, s, 2.0, "2w_even", "vix9d", 0.25, "r02c25", th), s, "r02c25", 2.0, "2w_even", "vix9d", g, th, 0.25))
        for sz in ("r01c10", "r05c25"):
            jobs.append(((k, s, 2.0, "2w_even", "vix9d", 0.10, sz, th), s, sz, 2.0, "2w_even", "vix9d", g, th, 0.10))
        for th2 in NB3.get(g, []):
            jobs.append(((k, s, 2.0, "2w_even", "vix9d", 0.10, "r02c25", th2), s, "r02c25", 2.0, "2w_even", "vix9d", g, th2, 0.10))
    print("option jobs", len(jobs), flush=True)
    with Pool(a.procs, initializer=init, initargs=("v3", 0.10)) as pool: R = pool.map(opt_job, jobs, chunksize=2)
    OR = {k: (res, tr) for k, res, tr in R}
    print("options done", round(time.time() - t0), flush=True)
    s0 = S0(dates); spy_r = rets(np.where(np.isfinite(D["tick"]["SPY"]["Cff"]), D["tick"]["SPY"]["Cff"], np.nan))
    EQ = {}; TR = []; comp = []; recon = []
    def addrow(fam, cid, label, eq_or_r, primary, extra=None, is_r=False, lo=SB.SIM_START):
        r = eq_or_r if is_r else rets(eq_or_r)
        m = metrics(r, spy_r, dates, lo, SB.END)
        row = {"family": fam, "config": cid, "label": label, "primary": primary, **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items()}, **(extra or {})}
        comp.append(row); EQ[cid] = r; return row
    # ---- recon + F1 ----
    for k, (res, tr) in OR.items():
        recon.append({"run": "|".join(map(str, k)), "recon_gap": res["recon_gap"], "n_trades": res["n_trades"], "day_trades_leg": res["day_trades_leg"],
                      "min_equity": res["min_equity"], "assign_risk_positions": res["stats"]["assign_risk_positions"], "entries": res["stats"]["entries"]})
    years = list(range(2015, 2027))
    def cagr_train(eq, Y):
        idx = [i for i in range(s0, Tn) if dates[i] < f"{Y}-01-01" and np.isfinite(eq[i])]
        return (eq[idx[-1]] / ACC) ** (252 / (idx[-1] - s0 + 1)) - 1
    f1sel = []
    def stitched(kind_src, pool_names, nworst, cm=2.0, tenor="2w_even", sk="c", itm=0.10, kind_trade="inv"):
        r = np.full(Tn, np.nan)
        for Y in years:
            sc = sorted(pool_names, key=lambda s: cagr_train(OR[("orig", s, 2.0, "2w_even", "c", 0.10)][0]["eq"], Y))[:nworst]
            f1sel.append({"rule": f"{kind_trade}_worst{nworst}", "cost": cm, "tenor": tenor, "scen": sk, "itm": itm, "test_year": Y, "selected_originals": ";".join(sc)})
            idx = [i for i in range(Tn) if dates[i][:4] == str(Y) and i >= s0]
            rr = np.nanmean([rets(OR[(kind_trade, s, cm, tenor, sk, itm)][0]["eq"].astype(float))[idx] for s in sc], axis=0)
            r[idx] = rr
        return r
    lo15 = "2015-01-02"
    for cm in (1.0, 2.0, 3.0):
        addrow("F1", f"F1a_c{cm:g}", "inverse of train-worst-1", stitched("inv", elig, 1, cm), cm == 2.0, {"cost": cm}, True, lo15)
        addrow("F1", f"F1b_c{cm:g}", "inverse of train-worst-3 blend", stitched("inv", elig, 3, cm), cm == 2.0, {"cost": cm}, True, lo15)
        addrow("F1", f"F1c_c{cm:g}", "opposite-signal of train-worst directional", stitched("opp", direc, 1, cm, kind_trade="opp"), cm == 2.0, {"cost": cm}, True, lo15)
    for tag, kw in [("odd", dict(tenor="2w_odd")), ("itm25", dict(itm=0.25)), ("iv100", dict(sk="iv100")), ("iv125", dict(sk="iv125"))]:
        addrow("F1", f"F1a_nb_{tag}", f"F1a sensitivity {tag}", stitched("inv", elig, 1, **kw), False, {"cost": 2.0}, True, lo15)
        addrow("F1", f"F1b_nb_{tag}", f"F1b sensitivity {tag}", stitched("inv", elig, 3, **kw), False, {"cost": 2.0}, True, lo15)
    addrow("F1", "F1c_nb_odd", "F1c sensitivity 2w_odd", stitched("opp", direc, 1, tenor="2w_odd", kind_trade="opp"), False, {"cost": 2.0}, True, lo15)
    # diagnostics table: gross vs net, original vs inverse
    diag = []
    for s in V.STRUCTS:
        row = {"structure": s, "inverse_allowed": s in elig}
        for cm in (0.0, 1.0, 2.0, 3.0):
            e = OR[("orig", s, cm, "2w_even", "c", 0.10)][0]["eq"].astype(float); row[f"orig_cagr_c{cm:g}"] = round((e[-1] / ACC) ** (252 / (Tn - s0)) - 1, 5)
            if s in elig:
                e = OR[("inv", s, cm, "2w_even", "c", 0.10)][0]["eq"].astype(float); row[f"inv_cagr_c{cm:g}"] = round((e[-1] / ACC) ** (252 / (Tn - s0)) - 1, 5)
        if s in elig:
            for sk in ("iv100", "iv125"):
                for kind in ("orig", "inv"):
                    e = OR[(kind, s, 2.0, "2w_even", sk, 0.10)][0]["eq"].astype(float); row[f"{kind}_cagr_c2_{sk}"] = round((e[-1] / ACC) ** (252 / (Tn - s0)) - 1, 5)
        diag.append(row)
    hind = min(elig, key=lambda s: diag[[d["structure"] for d in diag].index(s)]["orig_cagr_c2"])
    addrow("F1", "F1_HINDSIGHT_inv_worst", f"HINDSIGHT inverse of full-period worst ({hind})", OR[("inv", hind, 2.0, "2w_even", "c", 0.10)][0]["eq"].astype(float), False, {"cost": 2.0})
    # F1d bear shares, 10 phases
    phases = [(wd, p) for wd in range(5) for p in (0, 1)]
    for cm in (1.0, 2.0, 3.0):
        for wd, p in phases:
            if cm != 2.0 and (wd, p) != (0, 0): continue
            res = run_weights(bear_weights(D), biweekly_reb(dates, wd, p), cm, SB.TICKERS, D)
            prim = cm == 2.0 and (wd, p) == (0, 0)
            addrow("F1", f"F1d_c{cm:g}_wd{wd}p{p}", "long bear-trend tickers (opposite signal shares)", res["eq"], prim, {"cost": cm, "avg_exposure": round(res["avg_exposure"], 3)})
            if prim: TR += [{"config": "F1d_c2_wd0p0", **x} for x in res["trades"]]
    # momentum (original) shares phase for reference
    res = run_weights(lambda t: (lambda p: {s: min(0.5, 1 / len(p)) for s in p} if p else {})([s for s in SB.TICKERS if s in D["tick"] and D["tick"][s]["bull"][t - 1] and np.isfinite(D["tick"][s]["O"][t])]),
                      biweekly_reb(dates, 0, 0), 2.0, SB.TICKERS, D)
    addrow("REF", "REF_bull_shares_c2_wd0p0", "original bull-signal shares momentum (reference)", res["eq"], False, {"cost": 2.0})
    # ---- F2 ----
    for cm in (1.0, 2.0, 3.0):
        for nm, L, tg in [("F2a", 126, .10), ("F2b", 126, .15), ("F2c", 252, .10), ("F2d", 252, .15)]:
            res = run_weights(f2_weights(D, L, tg), month_reb(dates, 1), cm, ETFS, D)
            addrow("F2", f"{nm}_c{cm:g}", f"trend L{L} voltarget {tg}", res["eq"], cm == 2.0, {"cost": cm, "avg_exposure": round(res["avg_exposure"], 3), "turnover_per_yr": round(res["turnover_per_yr"], 3)})
            if cm == 2.0: TR += [{"config": f"{nm}_c2", **x} for x in res["trades"]]
    for nm, L, tg in [("F2a", 126, .10), ("F2b", 126, .15), ("F2c", 252, .10), ("F2d", 252, .15)]:
        for off in (8, 15, 22):
            res = run_weights(f2_weights(D, L, tg), month_reb(dates, off), 2.0, ETFS, D)
            addrow("F2", f"{nm}_nb_off{off}", f"{nm} offset {off}", res["eq"], False, {"cost": 2.0, "avg_exposure": round(res["avg_exposure"], 3)})
    for L in (189,):
        for tg in (0.10, 0.15):
            res = run_weights(f2_weights(D, L, tg), month_reb(dates, 1), 2.0, ETFS, D); addrow("F2", f"F2_nb_L{L}_t{tg}", "neighbor", res["eq"], False, {"cost": 2.0})
    for tg in (0.08, 0.12, 0.20):
        for L in (126, 252):
            res = run_weights(f2_weights(D, L, tg), month_reb(dates, 1), 2.0, ETFS, D); addrow("F2", f"F2_nb_L{L}_t{tg}", "neighbor", res["eq"], False, {"cost": 2.0})
    mr = month_reb(dates, 1)
    addrow("BENCH", "SPY_buyhold_100pct", "SPY buy-hold 100% (labelled exception)", run_weights(fixed_weights({"SPY_BENCH": 1.0}), {s0}, 2.0, ["SPY"], D)["eq"], False)
    addrow("BENCH", "SPY_50pct", "SPY 50% constant (monthly rebal)", run_weights(fixed_weights({"SPY": 0.5}), mr, 2.0, ["SPY"], D)["eq"], False)
    addrow("BENCH", "60_40", "SPY 50%/TLT 40% monthly (cap-compliant 60/40 analog: SPY capped at 50%)", run_weights(fixed_weights({"SPY": 0.5, "TLT": 0.4}), mr, 2.0, ["SPY", "TLT"], D)["eq"], False)
    addrow("BENCH", "EW6", "equal-weight 6 ETFs monthly", run_weights(fixed_weights({s: 1 / 6 for s in ETFS}), mr, 2.0, ETFS, D)["eq"], False)
    cr = np.full(Tn, (1 + CASH) ** (1 / 252) - 1); addrow("BENCH", "CASH", "cash 2%", cr, False, None, True)
    # ---- F3 ----
    for k, (s, g, th) in F3.items():
        for cm in (1.0, 2.0, 3.0):
            res, tr = OR[(k, s, cm, "2w_even", "vix9d", 0.10, "r02c25", th)]
            addrow("F3", f"{k}_c{cm:g}", f"{s} gate={g} thr={th} (VIX9D IV)", res["eq"].astype(float), cm == 2.0,
                   {"cost": cm, "n_trades": res["n_trades"], "assign_risk_positions": res["stats"]["assign_risk_positions"], "recon_gap": round(res["recon_gap"], 6)})
            if cm == 2.0: TR += [{"config": f"{k}_c2", **x} for x in tr]
        for tag, key in [("odd", (k, s, 2.0, "2w_odd", "vix9d", 0.10, "r02c25", th)), ("itm25", (k, s, 2.0, "2w_even", "vix9d", 0.25, "r02c25", th)),
                         ("r01c10", (k, s, 2.0, "2w_even", "vix9d", 0.10, "r01c10", th)), ("r05c25", (k, s, 2.0, "2w_even", "vix9d", 0.10, "r05c25", th))] + \
                        [(f"thr{t2}", (k, s, 2.0, "2w_even", "vix9d", 0.10, "r02c25", t2)) for t2 in NB3.get(g, [])]:
            res, _ = OR[key]; addrow("F3", f"{k}_nb_{tag}", f"{k} sensitivity {tag}", res["eq"].astype(float), False, {"cost": 2.0, "n_trades": res["n_trades"]})
    ratio = D["ratio"]
    def vrp_spy(th):
        def fn(t): return {"SPY": 0.5} if np.isfinite(ratio[t - 1]) and ratio[t - 1] >= th else {}
        return fn
    def chg(th):   # rebalance only when the gate state flips (state known at close t-1), plus the first session
        on = np.isfinite(ratio) & (ratio >= th)
        return {s0} | {t for t in range(s0 + 1, Tn) if on[t - 1] != on[t - 2]}
    for cm in (1.0, 2.0, 3.0):
        res = run_weights(vrp_spy(1.0), chg(1.0), cm, ["SPY"], D)
        addrow("F3", f"F3f_c{cm:g}", "SPY 50% when VIX9D/RV20>=1.0 else cash", res["eq"], cm == 2.0, {"cost": cm, "avg_exposure": round(res["avg_exposure"], 3), "turnover_per_yr": round(res["turnover_per_yr"], 2)})
    for th in (0.9, 1.1):
        res = run_weights(vrp_spy(th), chg(th), 2.0, ["SPY"], D); addrow("F3", f"F3f_nb_thr{th}", "neighbor", res["eq"], False, {"cost": 2.0, "avg_exposure": round(res["avg_exposure"], 3)})
    # ---- F4 ----
    for cm in (1.0, 2.0, 3.0):
        ins, outs = tom_sched(dates, -1, 3); res = run_weights(tom_weights(ins, outs, ["SPY", "QQQ", "IWM"]), ins | outs, cm, ["SPY", "QQQ", "IWM"], D)
        addrow("F4", f"F4a_c{cm:g}", "turn-of-month SPY/QQQ/IWM 1/3 each", res["eq"], cm == 2.0, {"cost": cm, "avg_exposure": round(res["avg_exposure"], 3)})
        if cm == 2.0: TR += [{"config": "F4a_c2", **x} for x in res["trades"]]
    for a_, b_ in [(-2, 3), (-1, 2), (-1, 4), (0, 3)]:
        ins, outs = tom_sched(dates, a_, b_); res = run_weights(tom_weights(ins, outs, ["SPY", "QQQ", "IWM"]), ins | outs, 2.0, ["SPY", "QQQ", "IWM"], D)
        addrow("F4", f"F4a_nb_{a_}_{b_}", "neighbor", res["eq"], False, {"cost": 2.0, "avg_exposure": round(res["avg_exposure"], 3)})
    # F4 placebo: random 5-session windows per month, same composition
    rng = np.random.default_rng(11); ins0, outs0 = tom_sched(dates, -1, 3); real = [r for r in comp if r["config"] == "F4a_c2"][0]["cagr"]; pl = []
    for _ in range(200):
        ins, outs = set(), set()
        months = sorted({d[:7] for d in dates})
        for mth in months:
            idx = [i for i, d in enumerate(dates) if d[:7] == mth]
            if len(idx) < 12: continue
            e = int(rng.integers(idx[0] + 4, idx[-1] - 6)); ins.add(e); outs.add(e + 4)
        res = run_weights(tom_weights(ins, outs, ["SPY", "QQQ", "IWM"]), ins | outs, 2.0, ["SPY", "QQQ", "IWM"], D)
        pl.append((res["eq"][-1] / ACC) ** (252 / (Tn - s0)) - 1)
    f4_placebo = {"n": len(pl), "real_cagr": real, "placebo_median": float(np.median(pl)), "placebo_p95": float(np.percentile(pl, 95)), "p_placebo_ge_real": float(np.mean(np.array(pl) >= real))}
    # ---- start-date sensitivity + pass rule ----
    alpha = 0.05 / 15
    byc = {r["config"]: r for r in comp}
    nbmap = {"F1a": ["F1a_nb_odd", "F1a_nb_itm25"], "F1b": ["F1b_nb_odd", "F1b_nb_itm25"], "F1c": ["F1c_nb_odd"],
             "F1d": [f"F1d_c2_wd{wd}p{p}" for wd, p in phases if (wd, p) != (0, 0)],
             **{k: [f"{k}_nb_off{o}" for o in (8, 15, 22)] for k in ("F2a", "F2b", "F2c", "F2d")},
             **{k: [c for c in byc if c.startswith(k + "_nb_")] for k in F3}, "F3f": ["F3f_nb_thr0.9", "F3f_nb_thr1.1"], "F4a": [c for c in byc if c.startswith("F4a_nb_")]}
    for r in comp:
        r["start_year_cagr"] = json.dumps({y: round(v, 4) for y, v in start_sens(EQ[r["config"]], dates, range(2012, 2023)).items()})
    for r in comp:
        if not r["primary"]: continue
        base = r["config"].rsplit("_c", 1)[0]
        c3 = byc.get(r["config"].replace("_c2", "_c3")) or {}
        ss = json.loads(r["start_year_cagr"]); frac_pos = np.mean([v > 0 for v in ss.values()]) if ss else float("nan")
        nb = [byc[c] for c in nbmap.get(base, []) if c in byc]
        nb_ok = all(n["excess_vs_vmspy_ann"] * r["excess_vs_vmspy_ann"] > 0 for n in nb) if nb else None
        stress_ok = None
        if base in ("F1a", "F1b"): stress_ok = all(byc[f"{base}_nb_{t}"]["cagr"] > 0 for t in ("itm25", "iv100"))
        if base in F3: stress_ok = byc[f"{base}_nb_itm25"]["cagr"] > 0
        crit = {"abs_c2_gt_cash": r["cagr"] > CASH, "abs_c3_gt_cash": (c3.get("cagr", -1) > CASH), "vmspy_sig": r["excess_vs_vmspy_ann"] > 0 and r["p_vs_vmspy"] < alpha,
                "start_years_2of3": frac_pos >= 2 / 3, "neighbors_same_sign": nb_ok, "stress_ok": stress_ok}
        r.update({f"crit_{k}": v for k, v in crit.items()}); r["frac_start_years_pos"] = round(float(frac_pos), 3)
        r["PASS"] = all(v is not False for v in crit.values()) and crit["vmspy_sig"]
        r["positive_absolute"] = r["cagr"] > 0
    # ---- write ----
    def W(name, rows):
        cols = list(dict.fromkeys(k for x in rows for k in x))
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, restval=""); w.writeheader(); [w.writerow(x) for x in rows]
    W("comparison.csv", comp); W("f1_inverse_diagnostics.csv", diag); W("f1_train_selection.csv", f1sel); W("recon.csv", recon); W("trades.csv", TR)
    keep = [r["config"] for r in comp if r["primary"] or r["family"] == "BENCH" or r["config"].startswith("REF") or "HINDSIGHT" in r["config"]]
    with (out / "equity.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["date"] + keep); eqs = {c: ACC * np.cumprod(np.nan_to_num(EQ[c], nan=0.0)[s0:] + 1) for c in keep}
        for j, t in enumerate(range(s0, Tn)): w.writerow([dates[t]] + [round(float(eqs[c][j]), 2) for c in keep])
    meta = {"run_id": rid, "run_at_et": run_at.isoformat(), "elapsed_s": round(time.time() - t0, 1), "hypotheses_sha256": hyp_sha, "fill_model": "v3", "itm_frac": 0.10,
            "n_option_runs": len(jobs), "n_comparison_rows": len(comp), "n_primary": sum(1 for r in comp if r["primary"]), "alpha_bonferroni": alpha,
            "inverse_eligible": elig, "directional": direc, "hindsight_worst": hind, "f4_placebo": f4_placebo,
            "max_abs_recon_gap": max(abs(x["recon_gap"]) for x in recon), "total_day_trades_leg": sum(x["day_trades_leg"] for x in recon),
            "vix9d_first": dates[int(np.argmax(np.isfinite(D["vix9d"])))], "sim": [SB.SIM_START, SB.END]}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    for r in comp:
        if r["primary"]: print(r["config"], r["final_equity"], round(r["cagr"], 4), round(r["max_dd"], 3), round(r["sharpe"], 2), "vsVM", round(r["excess_vs_vmspy_ann"], 4), "p", r["p_vs_vmspy"], "PASS", r.get("PASS"))


if __name__ == "__main__":
    main()
