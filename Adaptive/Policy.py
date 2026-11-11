"""Adaptive market-state policy study (2026-11-09). State known at close t-1 -> payoff family at open t.

Two pre-registered policy families (Adaptive/2026-11-09_adaptive-market-policy_hypotheses.json):
  A: transparent state/action table with 2-decision confirmation hysteresis
  B: per-arm ridge reward model refit at every decision on purged/embargoed past labels
Option prices are MODEL prices (BSM on VIX9D or trailing-RV IV, assumed skew, assumed spreads) -- not quotes.

Usage: .venv\\Scripts\\python -m Adaptive.Policy [--quick]   -> out/ap_<stamp>/
"""
from __future__ import annotations
import argparse, csv, json, math, hashlib, time, sys
from datetime import date, datetime
from pathlib import Path
import numpy as np

from Options import Validate10k as V
from Options import SynthBacktest as SB
from Options.Pricing import PayoffExtremes, Intrinsic
from Options.VolModel import IvScenario, ImpliedVol
from Options.DayTradeGuard import DayTradeGuard

ROOT = Path(__file__).resolve().parents[1]
HYP = ROOT / "Adaptive" / "2026-11-09_adaptive-market-policy_hypotheses.json"
BASE = "2026-11-09_adaptive-market-policy"
ACC = 10000.0
ARMS = ["cash", "equity", "defensive", "bull_call_debit", "bear_put_debit", "iron_condor"]
OPT_ARMS = ["bull_call_debit", "bear_put_debit", "iron_condor"]
FEATS = ["tr63", "sma200_gap", "rv20", "volchg", "dd252", "vix", "vrp", "term"]
Z99 = 2.33; MIN_TRAIN = 78; ALPHA = 20.0; CONF = 0.0025; HYST = 0.005
SLIP = SB.STOCK_SLIP_BPS / 1e4; CY = SB.CASH_YIELD


# ---------------------------------------------------------------- data / state
def LoadAll():
    import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
    D = V.LoadData2(); dates = D["dates"]; di = {d: i for i, d in enumerate(dates)}; Tn = len(dates)

    def series(path):
        a = np.full(Tn, np.nan); hist = []
        for b in json.loads(path.read_text())["bars"]:
            hist.append((b["date"], b["close"]))
            if b["date"] in di: a[di[b["date"]]] = b["close"]
        for i in range(1, Tn):
            if not np.isfinite(a[i]): a[i] = a[i - 1]
        return a, hist
    D["vix"], vh = series(ROOT / "data/vol_index/VIX.json")
    D["vix9d"], _ = series(ROOT / "data/vol_index/VIX9D.json")
    D["vix_hist_dates"] = np.array([h[0] for h in vh]); D["vix_hist"] = np.array([h[1] for h in vh], float)
    D["F"] = Features(D)
    return D


def Features(D):
    """F[name][i] uses only data dated <= i. Decisions at open t read F[...][t-1]."""
    tk = D["tick"]["SPY"]; C = tk["Cff"]; Tn = len(C); lr = np.r_[np.nan, np.diff(np.log(C))]
    F = {k: np.full(Tn, np.nan) for k in FEATS}
    for i in range(Tn):
        if i >= 63: F["tr63"][i] = C[i] / C[i - 63] - 1
        if i >= 199: F["sma200_gap"][i] = C[i] / np.mean(C[i - 199:i + 1]) - 1
        if i >= 61:
            rv20 = np.std(lr[i - 19:i + 1], ddof=1) * math.sqrt(252); rv60 = np.std(lr[i - 59:i + 1], ddof=1) * math.sqrt(252)
            F["rv20"][i] = rv20; F["volchg"][i] = math.log(rv20 / rv60)
        F["dd252"][i] = C[i] / np.max(C[max(0, i - 251):i + 1]) - 1
    F["vix"] = D["vix"].copy(); F["vrp"] = D["vix"] / 100.0 / F["rv20"]; F["term"] = D["vix9d"] / D["vix"]
    return F


def StateAt(D, t):
    return {k: float(D["F"][k][t - 1]) for k in FEATS}


# ---------------------------------------------------------------- packages (options) and labels
class Scen:
    def __init__(self, name, cost_mult=2.0, iv="vix9d", ivrp=1.0, itm=0.10, phase=0, B=0.10):
        self.name, self.cost_mult, self.iv, self.ivrp, self.itm, self.phase, self.B = name, cost_mult, iv, ivrp, itm, phase, B
        self.ivs = IvScenario(f"{iv}_x{ivrp:.2f}_sk10", ivrp, 0.10)

    def d(self): return {k: getattr(self, k) for k in ("name", "cost_mult", "iv", "ivrp", "itm", "phase", "B")}


def Periods(D, phase):
    ent = V.Entries(D["dates"], 2, phase); ts = sorted(t for t in ent if D["dates"][t] >= SB.SIM_START)
    out = []
    for j, t in enumerate(ts[:-1]):
        exp = ent[t]; x = exp - 1; nxt = ts[j + 1]
        if x > t and x < nxt and np.isfinite(D["F"]["sma200_gap"][t - 1]): out.append({"t": t, "exp": exp, "x": x, "next": nxt})
    return out


def BaseVol(D, sc, i):
    return D["vix9d"][i] / 100.0 if sc.iv == "vix9d" else D["tick"]["SPY"]["base"][i]


def Package(D, sc, P, arm):
    """Whole-package economics for one SPY option arm in period P. Open at raw open t (IV from close t-1), close at
    raw open x (IV from close x-1). Returns per-package $ cost (+ = pay), close proceeds, max loss, daily close marks."""
    V.FILL_MODEL, V.ITM_FRAC = "v3", sc.itm
    tk = D["tick"]["SPY"]; t, exp, x = P["t"], P["exp"], P["x"]; dates = D["dates"]
    S = tk["Or"][t]; Ty = SB.Tyears(dates, t, exp, True); Se = S - V.PvDivs(tk, t, exp, None)
    base = BaseVol(D, sc, t - 1)
    if not (np.isfinite(S) and np.isfinite(base)) or tk["splitf"][t] != tk["splitf"][exp]: return None
    atm = ImpliedVol(base, Se, Se, Ty, sc.ivs, 0.0, True, False)
    legs = V.Legs("SPY", arm, S, atm, Ty); mn, mx = PayoffExtremes(legs)
    mids = V.Mids("SPY", tk, legs, S, Se, base, Ty, sc.ivs, 0.0, True)
    cost = V.Fill("SPY", legs, mids, sc.cost_mult, True, S); max_loss = cost - mn * 100
    if not math.isfinite(mn) or max_loss <= 0 or (math.isfinite(mx) and cost > mx * 100): return None
    marks = {}
    for i in range(t, x):
        Si = tk["Cr"][i]; Ti = SB.Tyears(dates, i, exp, False); Sei = Si - V.PvDivs(tk, i, exp, None)
        m = V.Mids("SPY", tk, legs, Si, Sei, BaseVol(D, sc, i), Ti, sc.ivs, 0.0, False)
        marks[i] = 100 * SB.PackageMid(legs, m, Ti, mn, mx)
    Sx = tk["Or"][x]; Tx = SB.Tyears(dates, x, exp, True); Sex = Sx - V.PvDivs(tk, x, exp, None)
    mx_ = V.Mids("SPY", tk, legs, Sx, Sex, BaseVol(D, sc, x - 1), Tx, sc.ivs, 0.0, False)
    proc = -V.Fill("SPY", legs, mx_, sc.cost_mult, False, Sx)
    # assignment-risk flag (not simulated): short leg ITM with extrinsic < tick (or < next dividend for calls)
    arisk = False
    for i in range(t, x):
        Si = tk["Cr"][i]; nd = tk["div"][i + 1] if i + 1 <= exp else 0.0
        Ti = SB.Tyears(dates, i, exp, False); Sei = Si - V.PvDivs(tk, i, exp, None)
        m = V.Mids("SPY", tk, legs, Si, Sei, BaseVol(D, sc, i), Ti, sc.ivs, 0.0, False)
        for lg, mm in zip(legs, m):
            if lg["qty"] < 0:
                intr = Intrinsic(Si, lg["K"], lg["kind"])
                if intr > 0 and mm - intr < max(V.Tick("SPY", mm), nd if lg["kind"] == "C" else 0.0): arisk = True
    fee_open = sum(abs(lg["qty"]) for lg in legs) * SB.FEE_PER_CONTRACT * sc.cost_mult
    return {"legs": legs, "cost": cost, "proc": proc, "max_loss": max_loss, "marks": marks, "arisk": arisk, "fee_leg": fee_open,
            "mid_open": 100 * sum(lg["qty"] * m for lg, m in zip(legs, mids)), "mid_close": 100 * sum(lg["qty"] * m for lg, m in zip(legs, mx_))}


def ShareExposure(D, arm, t, B):
    """Ex-ante exposure from vol known at close t-1."""
    if arm == "equity":
        sig = 0.5 * (D["tick"]["SPY"]["base"][t - 1] + D["tick"]["QQQ"]["base"][t - 1]) * math.sqrt(10 / 252)
        e = min(1.0, B / (Z99 * sig)); return {"SPY": e / 2, "QQQ": e / 2}
    if arm == "defensive":
        sig = D["tick"]["TLT"]["base"][t - 1] * math.sqrt(10 / 252); return {"TLT": min(0.5, B / (Z99 * sig))}
    if arm == "spy_budget":
        sig = D["tick"]["SPY"]["base"][t - 1] * math.sqrt(10 / 252); return {"SPY": min(1.0, B / (Z99 * sig))}
    if arm == "spy_100": return {"SPY": 1.0}
    return {}


def Labels(D, sc, PS, PK):
    """Per-period fractional-sizing net return per arm + time it becomes known (index)."""
    L = np.zeros((len(PS), len(ARMS))); known = np.zeros((len(PS), len(ARMS)), int)
    for k, P in enumerate(PS):
        t, nx = P["t"], P["next"]; dt = (nx - t) / 252
        for a, arm in enumerate(ARMS):
            if arm == "cash": L[k, a] = CY * dt; known[k, a] = t
            elif arm in ("equity", "defensive"):
                w = ShareExposure(D, arm, t, sc.B); r = 0.0
                for s, e in w.items():
                    O = D["tick"][s]["O"]; r += e * (O[nx] / O[t] - 1 - 2 * sc.cost_mult * SLIP)
                L[k, a] = r + (1 - sum(w.values())) * CY * dt; known[k, a] = nx
            else:
                pk = PK[k].get(arm)
                L[k, a] = (sc.B * (pk["proc"] - pk["cost"]) / pk["max_loss"] if pk else 0.0) + CY * dt; known[k, a] = P["x"]
    return L, known


# ---------------------------------------------------------------- policies (return list of {arm: weight} per period)
def TableRec(D, PS, k):
    t = PS[k]["t"]; s = StateAt(D, t)
    m = D["vix_hist_dates"] < D["dates"][t]; vix_hi = s["vix"] >= np.quantile(D["vix_hist"][m], 0.70)
    past = D["F"]["vrp"][:t]; past = past[np.isfinite(past)]; rich = s["vrp"] >= np.median(past)
    up = s["sma200_gap"] > 0 and s["tr63"] > 0; dn = s["sma200_gap"] < 0 and s["tr63"] < 0
    if up: return ("bull_call_debit" if vix_hi else "equity"), {"vix_hi": bool(vix_hi), "rich": bool(rich), "trend": "up"}
    if dn: return ("bear_put_debit" if s["term"] > 1 else "defensive"), {"vix_hi": bool(vix_hi), "rich": bool(rich), "trend": "down"}
    return ("iron_condor" if (rich and not vix_hi) else "cash"), {"vix_hi": bool(vix_hi), "rich": bool(rich), "trend": "mixed"}


def PolicyTable(D, PS, K0):
    out, info = [], []; inc = "cash"; prev = None
    for k in range(K0, len(PS)):
        rec, st = TableRec(D, PS, k)
        if rec != inc and rec == prev: inc = rec
        prev = rec; out.append({inc: 1.0}); info.append({"rec": rec, **st})
    return out, info


def Usable(PS, known, j):
    """Rows k usable at decision j: k <= j-2 (embargo) and every arm label known strictly before t_j."""
    t = PS[j]["t"]; ks = [k for k in range(0, j - 1) if known[k].max() < t]
    return ks


def RidgeFit(X, Y, alpha=ALPHA):
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1; Z = (X - mu) / sd
    ym = Y.mean(0); A = Z.T @ Z + alpha * np.eye(Z.shape[1]); Bc = np.linalg.solve(A, Z.T @ (Y - ym))
    return mu, sd, ym, Bc


def PolicyRidge(D, PS, L, known, K0):
    out, info = [], []; inc = "cash"; ci = ARMS.index("cash")
    for j in range(K0, len(PS)):
        ks = Usable(PS, known, j)
        X = np.array([[StateAt(D, PS[k]["t"])[f] for f in FEATS] for k in ks]); Y = L[ks]
        mu, sd, ym, Bc = RidgeFit(X, Y)
        x = (np.array([StateAt(D, PS[j]["t"])[f] for f in FEATS]) - mu) / sd; pred = ym + x @ Bc
        a = int(np.argmax(pred)); cand = ARMS[a] if pred[a] - pred[ci] >= CONF else "cash"
        ii = ARMS.index(inc)
        if cand != inc and (pred[ARMS.index(cand)] - pred[ii] > HYST or pred[ii] < pred[ci]): inc = cand
        out.append({inc: 1.0}); info.append({"n_train": len(ks), "last_train_label_known": D["dates"][max(known[ks].max(1))],
                                             **{f"pred_{ARMS[i]}": float(pred[i]) for i in range(len(ARMS))}})
    return out, info


def PolicyAnnual(D, PS, L, known, K0):
    out = []; cur = "cash"; yr = None
    for j in range(K0, len(PS)):
        y = D["dates"][PS[j]["t"]][:4]
        if y != yr:
            yr = y; ks = Usable(PS, known, j); cur = ARMS[int(np.argmax(L[ks].mean(0)))]
        out.append({cur: 1.0})
    return out


# ---------------------------------------------------------------- simulator (whole contracts, real cash)
def Simulate(D, sc, PS, PK, K0, allocs, name, record=False):
    dates = D["dates"]; tick = D["tick"]; Tn = len(dates)
    t0 = PS[K0]["t"]; tend = PS[-1]["next"]
    cash = ACC; qty = {}; opts = []; eq = {}; guard = DayTradeGuard(dates)
    led = {"opt_pnl": 0.0, "interest": 0.0, "share_cost": 0.0, "opt_fees": 0.0, "opt_spread_cost": 0.0,
           "infeasible": 0, "opt_entries": 0, "arisk": 0, "turnover": 0.0, "share_cash_out": 0.0}
    trades = []; dec = {P["t"]: k for k, P in enumerate(PS) if k >= K0}; pid = 0
    prev_val = 0.0
    for t in range(t0, tend + 1):
        # options exits at open
        for p in list(opts):
            if p["x"] == t:
                pk = p["pk"]; cash += p["n"] * pk["proc"]; pnl = p["n"] * (pk["proc"] - pk["cost"]); led["opt_pnl"] += pnl
                led["opt_spread_cost"] += p["n"] * (pk["mid_close"] - pk["proc"] - pk["fee_leg"]); led["opt_fees"] += p["n"] * pk["fee_leg"]
                led["turnover"] += abs(p["n"] * pk["proc"])
                for q, lg in enumerate(pk["legs"]): guard.RecordClose(dates[t], f"{p['id']}:{q}", p["id"], p["n"] * abs(lg["qty"]))
                if record: trades.append({"policy": name, "date": dates[t], "action": "close", "instrument": p["arm"], "qty": p["n"],
                                          "price_pkg": round(pk["proc"], 2), "pnl": round(pnl, 2)})
                opts.remove(p)
        # share value at open (for P&L split)
        if t in dec:
            k = dec[t]; P = PS[k]; alloc = allocs[k - K0]
            E = eq.get(t - 1, ACC)
            # share targets
            target = {}
            for arm, w in alloc.items():
                if arm in ("equity", "defensive", "spy_budget", "spy_100"):
                    for s, e in ShareExposure(D, arm, t, sc.B).items(): target[s] = target.get(s, 0.0) + w * e * E / tick[s]["O"][t]
                if arm == "spy_fixedw":
                    target["SPY"] = target.get("SPY", 0.0) + w * E / tick["SPY"]["O"][t]
            for s in set(qty) | set(target):
                o = tick[s]["O"][t]; dq = target.get(s, 0.0) - qty.get(s, 0.0)
                if abs(dq) * o < 1.0: continue        # ignore sub-$1 drift rebalances
                c = abs(dq) * o * sc.cost_mult * SLIP; cash -= dq * o + c; led["share_cash_out"] += dq * o + c; led["share_cost"] += c; led["turnover"] += abs(dq) * o
                if record: trades.append({"policy": name, "date": dates[t], "action": "buy" if dq > 0 else "sell", "instrument": s, "qty": round(dq, 4),
                                          "price_pkg": round(o, 4), "pnl": round(-c, 2)})
                qty[s] = target.get(s, 0.0)
            qty = {s: q for s, q in qty.items() if q > 1e-12}
            for arm, w in alloc.items():
                if arm not in OPT_ARMS: continue
                pk = PK[k].get(arm)
                if pk is None: led["infeasible"] += 1; continue
                free = cash - sum(p["n"] * (p["pk"]["max_loss"] - p["pk"]["cost"]) for p in opts)
                n = int(min(w * sc.B * E, SB.TICKER_CAP * E, free) // pk["max_loss"])
                if n < 1: led["infeasible"] += 1; continue
                pid += 1; cash -= n * pk["cost"]; led["opt_entries"] += 1; led["arisk"] += int(pk["arisk"]); led["turnover"] += abs(n * pk["cost"])
                led["opt_spread_cost"] += n * (pk["cost"] - pk["mid_open"] - pk["fee_leg"])
                opts.append({"id": pid, "arm": arm, "n": n, "pk": pk, "x": P["x"], "t0": t})
                for q, lg in enumerate(pk["legs"]): guard.RecordOpen(dates[t], f"{pid}:{q}", pid, n * abs(lg["qty"]))
                if record: trades.append({"policy": name, "date": dates[t], "action": "open", "instrument": arm, "qty": n,
                                          "price_pkg": round(pk["cost"], 2), "pnl": 0.0,
                                          "legs": ";".join(f"{lg['qty']:+d}{lg['kind']}{lg['K']:g}" for lg in pk["legs"])})
        it = cash * CY / 252 if cash > 0 else 0.0; cash += it; led["interest"] += it
        sv = sum(q * tick[s]["Cff"][t] for s, q in qty.items())
        om = sum(p["n"] * p["pk"]["marks"][t] for p in opts)
        eq[t] = cash + sv + om
    guard_dt = len(guard.day_trades)
    led["day_trades"] = guard_dt
    sv_end = sum(q * tick[s]["Cff"][tend] for s, q in qty.items())
    open_m = sum(p["n"] * p["pk"]["marks"].get(tend, 0.0) for p in opts) - sum(p["n"] * p["pk"]["cost"] for p in opts)
    led["recon_gap"] = float(eq[tend] - (ACC + led["opt_pnl"] + led["interest"] + sv_end - led["share_cash_out"] + open_m))
    E = np.array([eq[t] for t in range(t0, tend + 1)])
    return {"name": name, "eq": E, "idx": list(range(t0, tend + 1)), "led": led, "trades": trades, "final_qty": qty, "open_opts": len(opts)}


def Metrics(r, D):
    E = np.r_[ACC, r["eq"]]; n = len(E) - 1; yrs = n / 252; ret = E[1:] / E[:-1] - 1
    dd = E / np.maximum.accumulate(E) - 1; vol = np.std(ret) * math.sqrt(252)
    cagr = (E[-1] / ACC) ** (1 / yrs) - 1
    return {"end": round(float(E[-1]), 2), "cagr": round(cagr, 4), "maxdd": round(float(dd.min()), 4), "vol": round(vol, 4),
            "sharpe_rf2": round((np.mean(ret) * 252 - CY) / vol, 3) if vol > 0 else float("nan"),
            "worst_2w": round(float(np.min(E[10:] / E[:-10] - 1)), 4) if n > 10 else float("nan")}


def PeriodLog(r, PS, K0):
    """Per-period log growth from close before decision t_k to close before t_{k+1}."""
    pos = {t: i for i, t in enumerate(r["idx"])}; E = r["eq"]
    val = lambda t: E[pos[t - 1]] if (t - 1) in pos else ACC
    return np.array([math.log(val(PS[k + 1]["t"]) / val(PS[k]["t"])) for k in range(K0, len(PS) - 1)])


def BlockBoot(x, reps=2000, block=6, seed=11):
    rng = np.random.default_rng(seed); n = len(x); m = []
    for _ in range(reps):
        idx = np.concatenate([np.arange(s, s + block) % n for s in rng.integers(0, n, n // block + 1)])[:n]; m.append(x[idx].mean())
    m = np.array(m); return float(np.mean(m <= 0)), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def CircularPerm(acts, L, K0):
    """acts: arm per period (k>=K0). Log-growth of fractional labels under actual vs all circular shifts of the action sequence."""
    a = np.array([ARMS.index(x) for x in acts]); Lw = L[K0:K0 + len(a)]
    f = lambda s: float(np.sum(np.log1p(Lw[np.arange(len(s)), s])))
    act = f(a); sh = [f(np.roll(a, s)) for s in range(1, len(a))]
    return act, float(np.mean(np.array(sh) >= act)), float(np.median(sh))


# ---------------------------------------------------------------- one scenario
def RunScenario(D, sc, record=False):
    PS = Periods(D, sc.phase)
    PK = [{arm: Package(D, sc, P, arm) for arm in OPT_ARMS} for P in PS]
    L, known = Labels(D, sc, PS, PK)
    K0 = next(j for j in range(len(PS)) if len(Usable(PS, known, j)) >= MIN_TRAIN)
    pa, ia = PolicyTable(D, PS, K0); pb, ib = PolicyRidge(D, PS, L, known, K0); pn = PolicyAnnual(D, PS, L, known, K0)
    nK = len(PS) - K0
    allocs = {"A_table": pa, "B_ridge": pb, "annual_selector": pn,
              **{f"fixed_{a}": [{a: 1.0}] * nK for a in ARMS},
              "fixed_mix": [{a: 0.2 for a in ARMS if a != "cash"}] * nK,
              "spy_cash_budget": [{"spy_budget": 1.0}] * nK, "spy_100_bench": [{"spy_100": 1.0}] * nK}
    orc = [{ARMS[int(np.argmax(L[k]))]: 1.0} for k in range(K0, len(PS))]; allocs["oracle_ILLUSTRATIVE"] = orc
    R = {n: Simulate(D, sc, PS, PK, K0, al, n, record) for n, al in allocs.items()}
    # supplementary ex-post same-vol SPY/cash for each policy (NOT tradable: weight uses full-sample realized vols)
    spy = R["spy_100_bench"]; sv = np.std(np.diff(np.log(spy["eq"])))
    for pname in ("A_table", "B_ridge"):
        w = min(1.0, np.std(np.diff(np.log(R[pname]["eq"]))) / sv)
        R[f"spy_samevol_EXPOST_{pname}"] = Simulate(D, sc, PS, PK, K0, [{"spy_fixedw": w}] * nK, f"spy_samevol_EXPOST_{pname}", record)
    return {"PS": PS, "PK": PK, "L": L, "known": known, "K0": K0, "allocs": allocs, "R": R, "info": {"A_table": ia, "B_ridge": ib}}


def Switches(al): 
    s = [next(iter(a)) for a in al]; return sum(1 for i in range(1, len(s)) if s[i] != s[i - 1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    t_start = time.time(); stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = ROOT / "out" / f"ap_{stamp}"; out.mkdir(parents=True)
    hyp_sha = hashlib.sha256(HYP.read_bytes()).hexdigest()
    D = LoadAll()
    scens = [Scen("central")]
    if not a.quick:
        scens += [Scen("cost0_tickonly", cost_mult=0.0), Scen("cost1x", cost_mult=1.0), Scen("cost3x", cost_mult=3.0),
                  Scen("iv_rv1.00", iv="rv", ivrp=1.0), Scen("iv_rv1.10", iv="rv", ivrp=1.1), Scen("iv_rv1.25", iv="rv", ivrp=1.25),
                  Scen("itm0.25", itm=0.25), Scen("phase1", phase=1), Scen("budget0.05", B=0.05)]
    comp, prim = [], []
    for sc in scens:
        res = RunScenario(D, sc, record=(sc.name == "central"))
        R, PS, K0, L = res["R"], res["PS"], res["K0"], res["L"]
        for n, r in R.items():
            m = Metrics(r, D); al = res["allocs"].get(n)
            comp.append({"scenario": sc.name, "series": n, "eval_start": D["dates"][r["idx"][0]], "eval_end": D["dates"][r["idx"][-1]], **m,
                         "switches": Switches(al) if al else "", **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r["led"].items()},
                         "turnover_x_per_yr": round(r["led"]["turnover"] / (len(r["eq"]) / 252) / float(np.mean(r["eq"])), 2)})
        # primary tests
        bench = {n: PeriodLog(R[n], PS, K0) for n in ("fixed_equity", "spy_cash_budget", "fixed_mix", "annual_selector")}
        for pname in ("A_table", "B_ridge"):
            pl = PeriodLog(R[pname], PS, K0); acts = [next(iter(x)) for x in res["allocs"][pname]]
            act, pperm, med = CircularPerm(acts, L, K0)
            row = {"scenario": sc.name, "policy": pname, "perm_actual_loggrowth": round(act, 4), "perm_median_shift": round(med, 4), "perm_p": round(pperm, 4)}
            for bn, bl in bench.items():
                p, lo, hi = BlockBoot(pl - bl); row[f"boot_p_vs_{bn}"] = round(p, 4); row[f"ann_logdiff_vs_{bn}"] = round(float(np.mean(pl - bl)) * 26, 4)
                row[f"ann_ci_vs_{bn}"] = f"[{lo * 26:.4f},{hi * 26:.4f}]"
            occ = {x: acts.count(x) for x in ARMS}; row.update({f"occ_{k}": v for k, v in occ.items()})
            prim.append(row)
        if sc.name == "central":
            cen = res
        print(f"{sc.name}: done {time.time() - t_start:.0f}s", flush=True)
    # ------------- write outputs
    W = lambda fn, rows: _w(out / f"{BASE}_{fn}.csv", rows)
    W("comparison", comp); W("primary_tests", prim)
    R, PS, K0, D_ = cen["R"], cen["PS"], cen["K0"], D
    eqrows = []
    for n, r in R.items():
        E = r["eq"]; dd = E / np.maximum.accumulate(E) - 1
        eqrows += [{"date": D["dates"][t], "series": n, "equity": round(float(e), 2), "drawdown_pct": round(float(x), 5)} for t, e, x in zip(r["idx"], E, dd)]
    W("equity", eqrows)
    alrows, trrows = [], []
    for pname in ("A_table", "B_ridge", "annual_selector"):
        prev = None
        for i, k in enumerate(range(K0, len(PS))):
            arm = next(iter(cen["allocs"][pname][i])); st = StateAt(D, PS[k]["t"])
            row = {"decision_date": D["dates"][PS[k]["t"]], "policy": pname, "arm": arm, **{f: round(v, 5) for f, v in st.items()},
                   **(cen["info"].get(pname, [{}] * 9999)[i] if pname in cen["info"] else {}),
                   "feasible_contracts": "" if arm not in OPT_ARMS else bool(cen["PK"][k].get(arm))}
            row = {kk: (round(vv, 6) if isinstance(vv, float) else vv) for kk, vv in row.items()}
            alrows.append(row)
            if prev is not None and arm != prev: trrows.append({"decision_date": row["decision_date"], "policy": pname, "from": prev, "to": arm, **{f: row[f] for f in FEATS}})
            prev = arm
    W("allocations", alrows); W("transitions", trrows)
    trades = [x for n in ("A_table", "B_ridge", "fixed_equity", "fixed_iron_condor") for x in R[n]["trades"]]
    W("trades", trades)
    lab = [{"decision_date": D["dates"][P["t"]], "known_by": D["dates"][max(cen["known"][k])], **{a: round(float(cen["L"][k, j]), 6) for j, a in enumerate(ARMS)}}
           for k, P in enumerate(PS)]
    W("arm_labels", lab)
    meta = {"base": BASE, "run": out.name, "created_et": datetime.now().isoformat(timespec="seconds"), "hypotheses_sha256": hyp_sha,
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "python": sys.version, "scenarios": [s.d() for s in scens],
            "n_periods": len(PS), "K0": K0, "eval_start": D["dates"][PS[K0]["t"]], "runtime_s": round(time.time() - t_start, 1),
            "costs": {"fee_per_contract_leg_side": SB.FEE_PER_CONTRACT, "half_spread_pct_SPY": SB.HALF_SPREAD_PCT["SPY"], "min_half_SPY": SB.MIN_HALF["SPY"],
                      "stock_slip_bps_per_side": SB.STOCK_SLIP_BPS, "fill_model": "v3: half = cost_mult*max(min_half, tick/2, pct*(extrinsic+itm_frac*intrinsic)); adverse tick rounding",
                      "cash_yield": CY, "rate": SB.RATE}}
    (out / f"{BASE}_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=1)); print("OUT", out)


def _w(path, rows):
    if not rows: path.write_text(""); return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
