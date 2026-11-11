"""$10k options-vs-shares validation run (v2 engine; v1 SynthBacktest left untouched for reproducibility).

Fixes vs v1 (see methodology .md):
- Options priced on UNADJUSTED (contract-consistent) spot; strikes on the raw grid; vol & signals still from the
  adjusted (total-return) series. Discrete dividends escrowed (S_eff = S - PV(divs ex in life)); covered-call stock
  legs are credited dividend cash on ex-dates. Cycles with a split inside the position life are EXCLUDED (counted).
- Fills rounded to real minimum ticks (buy up / sell down); half-spread >= half a tick. Short legs whose bid rounds
  to 0 receive 0.
- Sizing grid (predeclared): per-position max-loss fraction x aggregate open-max-loss cap; 50% ticker cap kept.
- Exposure: daily open max-loss / equity, entry-delta notional / equity, turnover, win rate, payoff ratio,
  assignment-risk flags (short ITM leg with extrinsic < tick, or < next dividend for calls). ASSIGNMENT IS STILL
  NOT SIMULATED: flagged positions are settled as if never assigned.
Usage: .venv\\Scripts\\python -m Options.Validate10k --procs 12 [--quick]
"""
from __future__ import annotations
import argparse, csv, json, math, time
from dataclasses import dataclass, asdict
from datetime import date, datetime
from multiprocessing import Pool
from pathlib import Path
import numpy as np

from Options import SynthBacktest as SB
from Options import Structures as ST
from Options.Pricing import BsPrice, BsGreeks, Intrinsic, PayoffExtremes
from Options.VolModel import IvScenario, ImpliedVol
from Options.DayTradeGuard import DayTradeGuard

BASE = "2026-11-09_options-validation"
CODE_VERSION = "stonk-data-0.5.0-optvalidate"
RAW = SB.ROOT / "data" / "daily_raw"
ACCOUNT = 10000.0
TICKS = {"SPY": (0.01, 0.01), "QQQ": (0.01, 0.01), "IWM": (0.01, 0.01), "AXTI": (0.05, 0.10), "QURE": (0.05, 0.10)}
TICK_DEFAULT = (0.01, 0.05)   # HOOD/SHOP observed on RH 2026-11-09; XBI/GDX/TLT assumed penny-pilot
ETF_STEP1 = {"SPY", "QQQ", "IWM"}
SIZING = {"r01c10": (0.01, 0.10), "r02c10": (0.02, 0.10), "r02c25": (0.02, 0.25), "r05c25": (0.05, 0.25)}
STRUCTS = ["long_call_atm", "long_put_atm", "bull_call_debit", "bear_put_debit", "bull_put_credit", "bear_call_credit",
           "iron_condor", "call_butterfly", "long_straddle", "covered_call", "cash_secured_put"]
EXITS = ["expiry", "tminus1"]
IVRPS = (1.0, 1.1, 1.25)
SCENARIOS = [IvScenario(f"iv{m:.2f}_sk10", ivrp=m, skew=0.10) for m in IVRPS]
CENTRAL = "iv1.10_sk10"; CEN_COST = 2.0
STRESS = [IvScenario("iv1.10_sk10_crush30", 1.1, 0.10, crush=0.30), IvScenario("iv1.10_sk10_earnjump", 1.1, 0.10, earn_mode="price_jump")]
TENORS = {"2w_even": (2, 0), "2w_odd": (2, 1), "4w_LONGER_TENOR_COMPARATOR": (4, 0)}
_D = {}


def Tick(sym, px):
    lo, hi = TICKS.get(sym, TICK_DEFAULT); return lo if px < 3 else hi


def RoundBuy(px, tk): return math.ceil(px / tk - 1e-9) * tk


def RoundSell(px, tk): return max(math.floor(px / tk + 1e-9) * tk, 0.0)


def Entries(dates, period, phase):
    wo = np.array([(date.fromisoformat(d) - date(2000, 1, 3)).days // 7 for d in dates]); out = {}
    for t in range(1, len(dates)):
        if wo[t] != wo[t - 1] and wo[t] % period == phase:
            nxt = np.where(wo == wo[t] + period - 1)[0]
            if len(nxt) and nxt[-1] > t: out[t] = int(nxt[-1])
    return out


def LoadData2(end=SB.END):
    D = SB.LoadData(end); dates = D["dates"]; di = {d: i for i, d in enumerate(dates)}; Tn = len(dates)
    for s, tk in D["tick"].items():
        js = json.loads((RAW / f"{s}.json").read_text())
        Or = np.full(Tn, np.nan); Cr = np.full(Tn, np.nan); sf = np.full(Tn, np.nan)
        for b in js["bars"]:
            i = di.get(b["date"])
            if i is not None: Or[i], Cr[i], sf[i] = b["open"], b["close"], b["split_factor_after"]
        for i in range(1, Tn):
            if not np.isfinite(Cr[i]): Cr[i] = Cr[i - 1]
            if not np.isfinite(sf[i]): sf[i] = sf[i - 1]
        div = np.zeros(Tn)
        for d, v in js["dividends"]:
            i = di.get(d)
            if i is not None: div[i] += v
        tk.update({"Or": Or, "Cr": Cr, "splitf": sf, "div": div, "splits": js["splits"]})
    D["entries_by_tenor"] = {k: Entries(dates, *v) for k, v in TENORS.items()}
    return D


def _init():
    import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
    _D.update(LoadData2())


def _step(sym):
    return (lambda S: 1.0) if sym in ETF_STEP1 else SB_STEP


SB_STEP = ST.StrikeStep


def Bias(name):
    """inv:<s> = literal inverse (every leg's qty negated, same entry signal); opp:<s> = same legs, opposite signal."""
    if name.startswith("inv:"): return ST.STRUCTURES[name[4:]][0]
    if name.startswith("opp:"): return {"bull": "bear", "bear": "bull"}.get(ST.STRUCTURES[name[4:]][0], "none")
    return ST.STRUCTURES[name][0]


def Legs(sym, name, S, atm, T):
    base = name[4:] if name[:4] in ("inv:", "opp:") else name
    ST.StrikeStep = _step(sym)
    try: legs = ST.BuildLegs(base, S, atm, T)
    finally: ST.StrikeStep = SB_STEP
    if name.startswith("inv:"):
        legs = [{**lg, "qty": -lg["qty"]} for lg in legs]
    return legs


def InverseAllowed(name):
    """defined-risk only, and no short stock in a cash account."""
    legs = Legs("SPY", "inv:" + name, 500.0, 0.2, 10 / 365)
    if any(lg["kind"] == "S" and lg["qty"] < 0 for lg in legs): return False
    return math.isfinite(PayoffExtremes(legs)[0])


def PvDivs(tk, t, exp, Ty_per_day):
    idx = np.nonzero(tk["div"][t + 1:exp + 1])[0]
    return sum(tk["div"][t + 1 + j] * math.exp(-SB.RATE * (j + 1) / 252) for j in idx)


def Mids(sym, tk, legs, S_raw, S_eff, base, T, scen, ev, at_entry):
    out = []
    for lg in legs:
        if lg["kind"] == "S": out.append(S_raw); continue
        iv = ImpliedVol(base, S_eff, lg["K"], T, scen, ev, at_entry, sym in SB.EVENT_NAMES)
        out.append(BsPrice(S_eff, lg["K"], T, SB.RATE, 0.0, iv, lg["kind"]))
    return out


FILL_MODEL = "v2"   # "v2" = original (pct x full mid); "v3" = repaired (2026-11-09 abstract-edge-search)
ITM_FRAC = 0.10     # v3: pct half-spread applies to extrinsic + ITM_FRAC x intrinsic (stress 0.25, optimistic 0.0)


def LegHalfSpread(sym, m, intr, cost_mult):
    tk = Tick(sym, m)
    if FILL_MODEL == "v2":
        base = SB.HALF_SPREAD_PCT[sym] * m
    else:
        base = SB.HALF_SPREAD_PCT[sym] * (max(m - intr, 0.0) + ITM_FRAC * intr)
    return cost_mult * max(SB.MIN_HALF.get(sym, SB.MIN_HALF_DEFAULT), base, tk / 2)


def Fill(sym, legs, mids, cost_mult, opening, S=None):
    """$ cash flow per package (positive = we pay). v3 needs S (raw spot) for intrinsic: half-spread scales with
    extrinsic + ITM_FRAC*intrinsic, and selling a long American leg never fills below intrinsic minus the stock-exit
    cost (exercise-and-sell floor). Fees are always charged per contract (cost_mult*$0.03), both directions."""
    flow = fees = 0.0
    for lg, m in zip(legs, mids):
        q = lg["qty"] if opening else -lg["qty"]
        if lg["kind"] == "S":
            flow += q * m * (1 + cost_mult * SB.STOCK_SLIP_BPS / 1e4 * (1 if q > 0 else -1)) * 100; continue
        intr = Intrinsic(S, lg["K"], lg["kind"]) if (S is not None and FILL_MODEL != "v2") else 0.0
        hs = LegHalfSpread(sym, m, intr, cost_mult)
        if q > 0:
            px = RoundBuy(m + hs, Tick(sym, m + hs))
        else:
            px = RoundSell(m - hs, Tick(sym, max(m - hs, 0)))
            if FILL_MODEL != "v2" and intr > 0:
                floor = intr - cost_mult * SB.STOCK_SLIP_BPS / 1e4 * S - SB.FEE_PER_CONTRACT * cost_mult / 100
                px = max(px, RoundSell(floor, Tick(sym, max(floor, 0))))
        flow += q * px * 100; fees += abs(q) * SB.FEE_PER_CONTRACT * cost_mult
    return flow + fees


@dataclass
class Cfg2:
    structure: str
    exit: str
    sizing: str
    scen: IvScenario
    cost_mult: float
    tenor: str = "2w_even"
    record: bool = False


def RunOptions2(cfg: Cfg2, D=None):
    D = D or _D; dates, tick = D["dates"], D["tick"]; entries = D["entries_by_tenor"][cfg.tenor]
    Tn = len(dates); s0 = next(i for i, d in enumerate(dates) if d >= SB.SIM_START)
    bias = Bias(cfg.structure); rf, cap = SIZING[cfg.sizing]; scen = cfg.scen
    cash = ACCOUNT; eq = np.full(Tn, np.nan); risk_x = np.zeros(Tn); delta_x = np.zeros(Tn)
    positions, trades = [], []; last_open = 0.0; guard = DayTradeGuard(dates); pid = 0; flows = 0.0
    st = {"entries": 0, "infeasible": 0, "split_excluded": 0, "earn_skip": 0, "unlimited_reject": 0, "bad_pricing": 0,
          "assign_risk_positions": 0, "assign_risk_days": 0, "div_credits": 0, "interest": 0.0}

    def close(p, t, proc, how):
        nonlocal cash, flows
        cash += p["n"] * proc; flows += abs(p["n"] * proc)
        for k, lg in enumerate(p["legs"]): guard.RecordClose(dates[t], f"{p['id']}:{k}", p["id"], p["n"] * abs(lg["qty"]))
        trades.append({"entry_date": dates[p["t0"]], "exit_date": dates[t], "symbol": p["sym"], "structure": cfg.structure, "qty": p["n"],
                       "entry_cost_pkg": round(p["cost"], 2), "exit_proceeds_pkg": round(proc, 2), "pnl": round(p["n"] * (proc - p["cost"]) + p["divs"], 2),
                       "max_loss": round(p["max_loss"], 2), "exit_how": how, "assign_risk_flag": p["arisk"],
                       "legs": ";".join(f"{lg['qty']:+d}{lg['kind']}{lg['K']:g}" for lg in p["legs"]), "expiry": dates[p["exp"]]})

    for t in range(s0, Tn):
        for p in list(positions):
            if p["exit_at"] == t:
                tk = tick[p["sym"]]; S = tk["Or"][t] if np.isfinite(tk["Or"][t]) else tk["Cr"][t - 1]
                Ty = SB.Tyears(dates, t, p["exp"], True); Se = S - PvDivs(tk, t, p["exp"], None)
                mids = Mids(p["sym"], tk, p["legs"], S, Se, tk["base"][t - 1], Ty, scen, SB.EventVar(tk, t, p["exp"], scen), False)
                close(p, t, -Fill(p["sym"], p["legs"], mids, cfg.cost_mult, False, S), "open_exit"); positions.remove(p)
        if t in entries:
            exp = entries[t]; E = eq[t - 1] if np.isfinite(eq[t - 1]) else cash; held = {p["sym"] for p in positions}
            for sym in SB.TICKERS:
                if sym not in tick or sym in held: continue
                tk = tick[sym]
                if not (np.isfinite(tk["Or"][t]) and np.isfinite(tk["O"][t]) and np.isfinite(tk["base"][t - 1])): continue
                if bias == "bull" and not tk["bull"][t - 1]: continue
                if bias == "bear" and not tk["bear"][t - 1]: continue
                if bias == "none": continue
                if cfg.structure.startswith("inv:") and any(lg["kind"] == "S" for lg in Legs(sym, cfg.structure, 100.0, 0.3, 0.04)): st["unlimited_reject"] += 1; continue
                exit_at = -1 if cfg.exit == "expiry" else exp - 1
                if cfg.exit != "expiry" and exit_at <= t: continue
                if tk["splitf"][t] != tk["splitf"][exp]: st["split_excluded"] += 1; continue
                S = tk["Or"][t]; Ty = SB.Tyears(dates, t, exp, True); Se = S - PvDivs(tk, t, exp, None)
                ev = SB.EventVar(tk, t, exp, scen)
                atm = ImpliedVol(tk["base"][t - 1], Se, Se, Ty, scen, 0.0, True, sym in SB.EVENT_NAMES)
                legs = Legs(sym, cfg.structure, S, atm, Ty); mn, mx = PayoffExtremes(legs)
                if not math.isfinite(mn): st["unlimited_reject"] += 1; continue
                mids = Mids(sym, tk, legs, S, Se, tk["base"][t - 1], Ty, scen, ev, True)
                cost = Fill(sym, legs, mids, cfg.cost_mult, True, S); max_loss = cost - mn * 100
                if max_loss <= 0 or (math.isfinite(mx) and cost > mx * 100): st["bad_pricing"] += 1; continue
                free = cash - sum(p["reserve"] for p in positions); open_risk = sum(p["max_loss"] for p in positions)
                n = int(min(rf * E, SB.TICKER_CAP * E, free, cap * E - open_risk) // max_loss)
                if n < 1: st["infeasible"] += 1; continue
                pid += 1; cash -= n * cost; flows += abs(n * cost)
                dl = 0.0
                for lg in legs:
                    if lg["kind"] == "S": dl += lg["qty"]; continue
                    iv = ImpliedVol(tk["base"][t - 1], Se, lg["K"], Ty, scen, ev, True, sym in SB.EVENT_NAMES)
                    dl += lg["qty"] * BsGreeks(Se, lg["K"], Ty, SB.RATE, 0.0, iv, lg["kind"])["delta"]
                p = {"id": pid, "sym": sym, "legs": legs, "n": n, "t0": t, "exp": exp, "exit_at": exit_at, "cost": cost, "max_loss": n * max_loss,
                     "reserve": n * (max_loss - cost), "mn": mn, "mx": mx, "delta_sh": dl * 100 * n, "divs": 0.0, "arisk": False}
                positions.append(p); st["entries"] += 1
                for k, lg in enumerate(legs): guard.RecordOpen(dates[t], f"{pid}:{k}", pid, n * abs(lg["qty"]))
        mark = 0.0
        for p in list(positions):
            tk = tick[p["sym"]]; S = tk["Cr"][t]
            stock_q = sum(lg["qty"] for lg in p["legs"] if lg["kind"] == "S")
            if stock_q and tk["div"][t] > 0 and t > p["t0"]:
                c = p["n"] * stock_q * 100 * tk["div"][t]; cash += c; p["divs"] += c; st["div_credits"] += 1
            if cfg.exit == "expiry" and t == p["exp"]:
                proc = sum(lg["qty"] * (S if lg["kind"] == "S" else Intrinsic(S, lg["K"], lg["kind"])) for lg in p["legs"]) * 100
                close(p, t, proc, "expiry_settle"); positions.remove(p); continue
            Ty = SB.Tyears(dates, t, p["exp"], False); Se = S - PvDivs(tk, t, p["exp"], None)
            mids = Mids(p["sym"], tk, p["legs"], S, Se, tk["base"][t], Ty, scen, SB.EventVar(tk, t, p["exp"], scen), False)
            v = p["n"] * 100 * SB.PackageMid(p["legs"], mids, Ty, p["mn"], p["mx"]); mark += v
            nd = tk["div"][t + 1] if t + 1 < Tn and t + 1 <= p["exp"] else 0.0
            for lg, m in zip(p["legs"], mids):
                if lg["kind"] != "S" and lg["qty"] < 0:
                    intr = Intrinsic(S, lg["K"], lg["kind"])
                    if intr > 0 and (m - intr) < max(Tick(p["sym"], m), nd if lg["kind"] == "C" else 0.0):
                        st["assign_risk_days"] += 1
                        if not p["arisk"]: p["arisk"] = True; st["assign_risk_positions"] += 1
        it = cash * SB.CASH_YIELD / 252; cash += it; st["interest"] += it
        eq[t] = cash + mark; last_open = mark - sum(p["n"] * p["cost"] for p in positions) + sum(p["divs"] for p in positions)
        if eq[t] > 0:
            risk_x[t] = sum(p["max_loss"] for p in positions) / eq[t]
            delta_x[t] = sum(abs(p["delta_sh"]) * tick[p["sym"]]["Cr"][t] for p in positions) / eq[t]
    pn = np.array([x["pnl"] for x in trades]) if trades else np.array([])
    wins, losses = pn[pn > 0], pn[pn <= 0]
    seg = slice(s0, Tn); yrs = (Tn - s0) / 252; me = float(np.nanmean(eq[seg]))
    res = {"eq": eq.astype(np.float32), "stats": st, "n_trades": len(trades), "day_trades_leg": len(guard.day_trades),
           "win_rate": float(np.mean(pn > 0)) if len(pn) else float("nan"),
           "payoff_ratio": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) and losses.mean() < 0 else float("nan"),
           "avg_risk_exposure": float(risk_x[seg].mean()), "peak_risk_exposure": float(risk_x[seg].max()),
           "avg_delta_exposure": float(delta_x[seg].mean()), "peak_delta_exposure": float(delta_x[seg].max()),
           "turnover_per_yr": float(flows / yrs / me) if me > 0 else float("nan"),
           "min_equity": float(np.nanmin(eq[seg])), "severe_loss_lt50": bool(np.nanmin(eq[seg]) < 0.5 * ACCOUNT), "ruin_lt10": bool(np.nanmin(eq[seg]) < 0.1 * ACCOUNT),
           "recon_gap": float(eq[-1] - (ACCOUNT + pn.sum() + st["interest"] + last_open))}
    if cfg.record: res["trades"] = trades
    return res


def RunSharesPhase(cost_mult, weekday, parity, D=None):
    """Shares momentum (adjusted/total-return series, fractional shares) rebalanced at the first session on/after
    `weekday` in weeks with week-ordinal % 2 == parity. Signals from close t-1, fill at open t."""
    D = D or _D; dates, tick = D["dates"], D["tick"]; Tn = len(dates); s0 = next(i for i, d in enumerate(dates) if d >= SB.SIM_START)
    wo = [(date.fromisoformat(d) - date(2000, 1, 3)).days // 7 for d in dates]; wd = [date.fromisoformat(d).weekday() for d in dates]
    reb, done = set(), set()
    for t in range(1, Tn):
        if wo[t] % 2 == parity and wd[t] >= weekday and wo[t] not in done: reb.add(t); done.add(wo[t])
    cash = ACCOUNT; qty = {}; eq = np.full(Tn, np.nan); turn = 0.0; expo = []
    for t in range(s0, Tn):
        if t in reb:
            E = cash + sum(q * tick[s]["Cff"][t - 1] for s, q in qty.items())
            picks = [s for s in SB.TICKERS if s in tick and tick[s]["bull"][t - 1] and np.isfinite(tick[s]["O"][t])]
            w = min(0.5, 1.0 / len(picks)) if picks else 0.0
            target = {s: w * E / tick[s]["O"][t] for s in picks}
            for s in set(qty) | set(target):
                o = tick[s]["O"][t] if np.isfinite(tick[s]["O"][t]) else tick[s]["Cff"][t - 1]
                dq = target.get(s, 0.0) - qty.get(s, 0.0); cash -= dq * o + abs(dq) * o * cost_mult * SB.STOCK_SLIP_BPS / 1e4
                turn += abs(dq) * o; qty[s] = target.get(s, 0.0)
            qty = {s: q for s, q in qty.items() if q > 0}
        cash += cash * SB.CASH_YIELD / 252 if cash > 0 else 0.0
        inv = sum(q * tick[s]["Cff"][t] for s, q in qty.items()); eq[t] = cash + inv; expo.append(inv / eq[t] if eq[t] > 0 else 0)
    return {"eq": eq.astype(np.float32), "avg_exposure": float(np.mean(expo)), "peak_exposure": float(np.max(expo)),
            "turnover_per_yr": float(turn / ((Tn - s0) / 252) / np.nanmean(eq[s0:]))}


def _initfill(fm, itm):
    global FILL_MODEL, ITM_FRAC
    FILL_MODEL, ITM_FRAC = fm, itm; _init()


def _job(a):
    s, x, z, sc, cm, ten = a
    return a, RunOptions2(Cfg2(s, x, z, sc, cm, ten))


def _w(path, rows, cols=None):
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in rows: w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=12); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fill", default="v2", choices=["v2", "v3"]); ap.add_argument("--itm-frac", type=float, default=0.10)
    args = ap.parse_args(); t0 = time.time(); run_at = datetime.now(SB.ET)
    global FILL_MODEL, ITM_FRAC
    FILL_MODEL, ITM_FRAC = args.fill, args.itm_frac
    run_id = f"ov_{run_at.strftime('%Y%m%dT%H%M%S')}" + ("_quick" if args.quick else "") + (f"_fill{args.fill}_itm{args.itm_frac:g}" if args.fill != "v2" else "")
    rd = SB.OUT / run_id; rd.mkdir(parents=True, exist_ok=False)
    structs, exits, sizing, scens, cms = STRUCTS, EXITS, list(SIZING), SCENARIOS, [1.0, 2.0, 3.0]
    if args.quick: structs, sizing, scens, cms = ["bull_call_debit", "iron_condor", "covered_call"], ["r02c10"], SCENARIOS[1:2], [2.0]
    cen = next(s for s in SCENARIOS if s.name == CENTRAL)
    jobs = [(s, x, z, sc, cm, "2w_even") for s in structs for x in exits for z in sizing for sc in scens for cm in cms]
    if not args.quick:
        jobs += [(s, x, z, sc, CEN_COST, "2w_even") for s in structs for x in exits for z in sizing for sc in STRESS]
        jobs += [(s, x, z, cen, CEN_COST, ten) for s in structs for x in exits for z in sizing for ten in ("2w_odd", "4w_LONGER_TENOR_COMPARATOR")]
    (rd / "jobs_predeclared.json").write_text(json.dumps([[a[0], a[1], a[2], a[3].name, a[4], a[5]] for a in jobs]))
    print(f"{run_id}: {len(jobs)} option runs (predeclared list saved)", flush=True)
    _init()
    with Pool(args.procs, initializer=_initfill, initargs=(FILL_MODEL, ITM_FRAC)) as pool: res = pool.map(_job, jobs, chunksize=2)
    print(f"grid {time.time()-t0:.0f}s", flush=True)
    D = _D; dates = D["dates"]; W = {k: SB.WinIdx(dates, *v) for k, v in SB.WINDOWS.items()}
    s0 = next(i for i, d in enumerate(dates) if d >= SB.SIM_START)
    spy = D["tick"]["SPY"]["Cff"].astype(float); spy_r = spy[1:] / spy[:-1] - 1
    spy_vol = float(np.nanstd(spy_r[s0:], ddof=1) * math.sqrt(252))
    reg, curves = [], {}
    for (s, x, z, sc, cm, ten), r in res:
        m = {w: SB.Metrics(r["eq"], *W[w]) for w in W}
        wv = min(1.0, (m["ALL"]["vol"] or 0) / spy_vol) if m["ALL"].get("vol") == m["ALL"].get("vol") else 0.0
        vm = np.full(len(dates), np.nan); vm[s0 - 1] = ACCOUNT
        for t in range(s0, len(dates)): vm[t] = vm[t - 1] * (1 + wv * spy_r[t - 1] + (1 - wv) * SB.CASH_YIELD / 252)
        mv = SB.Metrics(vm, *W["ALL"])
        row = {"variant": f"{s}|{x}|{z}", "structure": s, "exit": x, "sizing": z, "tenor": ten, "scenario": sc.name, "ivrp": sc.ivrp, "crush": sc.crush,
               "earn_mode": sc.earn_mode, "cost_mult": cm, "account": ACCOUNT, "n_trades": r["n_trades"], "win_rate": r["win_rate"], "payoff_ratio": r["payoff_ratio"],
               "day_trades_leg": r["day_trades_leg"], **r["stats"], **{k: r[k] for k in ("avg_risk_exposure", "peak_risk_exposure", "avg_delta_exposure",
               "peak_delta_exposure", "turnover_per_yr", "min_equity", "severe_loss_lt50", "ruin_lt10", "recon_gap")}, "final_equity": float(r["eq"][-1]),
               "volmatched_spy_weight": wv, "volmatched_spy_ALL_cagr": mv["cagr"], "volmatched_spy_ALL_sharpe": mv["sharpe"]}
        for w in W:
            for k in ("cagr", "sharpe", "maxdd", "vol"): row[f"{w}_{k}"] = m[w].get(k)
        reg.append(row); curves[(s, x, z, sc.name, cm, ten)] = r["eq"]
    # shares: all 10 weekday x parity phases x 3 costs; cash; SPY
    sh = []; sh_curves = {}
    for wd in range(5):
        for par in (0, 1):
            for cm in cms:
                r = RunSharesPhase(cm, wd, par); m = {w: SB.Metrics(r["eq"], *W[w]) for w in W}
                row = {"variant": f"shares_momentum|wd{wd}|p{par}", "weekday": wd, "parity": par, "cost_mult": cm, "account": ACCOUNT,
                       **{k: r[k] for k in ("avg_exposure", "peak_exposure", "turnover_per_yr")}, "min_equity": float(np.nanmin(r["eq"][s0:])),
                       "severe_loss_lt50": bool(np.nanmin(r["eq"][s0:]) < 5000), "ruin_lt10": bool(np.nanmin(r["eq"][s0:]) < 1000)}
                for w in W:
                    for k in ("cagr", "sharpe", "maxdd", "vol"): row[f"{w}_{k}"] = m[w].get(k)
                sh.append(row); sh_curves[(wd, par, cm)] = r["eq"]
    _w(rd / f"{BASE}_shares-phases.csv", sh)
    spy_eq = np.full(len(dates), np.nan); spy_eq[s0 - 1:] = spy[s0 - 1:] / spy[s0 - 1] * ACCOUNT
    cash_eq = SB.RunCash(ACCOUNT, D)["eq"]
    for nm, e in (("SPY_buyhold_totalreturn", spy_eq), ("cash_2pct", cash_eq)):
        m = {w: SB.Metrics(e, *W[w]) for w in W}
        reg.append({"variant": nm, "structure": nm, "account": ACCOUNT, **{f"{w}_{k}": m[w].get(k) for w in W for k in ("cagr", "sharpe", "maxdd", "vol")}})
    _w(rd / f"{BASE}_registry.csv", reg)
    # summary per variant (2w_even core grid)
    grid = {sc.name for sc in scens}; summ = []
    for s in structs:
        for x in exits:
            for z in sizing:
                core = [r for r in reg if r.get("structure") == s and r.get("exit") == x and r.get("sizing") == z and r.get("tenor") == "2w_even" and r["scenario"] in grid]
                if not core: continue
                c = next((r for r in core if r["scenario"] == CENTRAL and r["cost_mult"] == CEN_COST), core[0])
                oth = {(r["tenor"], r["scenario"]): r for r in reg if r.get("structure") == s and r.get("exit") == x and r.get("sizing") == z and r not in core}
                pre = [r["PRE_sharpe"] if r["PRE_sharpe"] == r["PRE_sharpe"] else -9.0 for r in core]
                summ.append({"variant": f"{s}|{x}|{z}", "structure": s, "exit": x, "sizing": z, "PRE_min_sharpe": min(pre), "PRE_median_sharpe": float(np.median(pre)),
                             "PRE_frac_cagr_pos": float(np.mean([r["PRE_cagr"] > 0 for r in core])), "worst_ALL_maxdd": min(r["ALL_maxdd"] for r in core),
                             "any_severe_loss": any(r["severe_loss_lt50"] for r in core), "any_ruin": any(r["ruin_lt10"] for r in core),
                             **{f"central_{w}_{k}": c[f"{w}_{k}"] for w in ("EARLY", "IS", "HO", "ALL") for k in ("cagr", "sharpe", "maxdd")},
                             **{f"central_{k}": c[k] for k in ("n_trades", "win_rate", "payoff_ratio", "avg_risk_exposure", "peak_risk_exposure", "avg_delta_exposure",
                                "peak_delta_exposure", "turnover_per_yr", "infeasible", "split_excluded", "assign_risk_positions", "volmatched_spy_weight",
                                "volmatched_spy_ALL_cagr", "volmatched_spy_ALL_sharpe")},
                             "ivrp1.00_cost1_ALL_cagr": next((r["ALL_cagr"] for r in core if r["ivrp"] == 1.0 and r["cost_mult"] == 1.0), None),
                             "ivrp1.25_cost1_ALL_cagr": next((r["ALL_cagr"] for r in core if r["ivrp"] == 1.25 and r["cost_mult"] == 1.0), None),
                             "cost3_central_ALL_cagr": next((r["ALL_cagr"] for r in core if r["scenario"] == CENTRAL and r["cost_mult"] == 3.0), None),
                             "stress_crush30_ALL_cagr": (oth.get(("2w_even", "iv1.10_sk10_crush30")) or {}).get("ALL_cagr"),
                             "stress_earnjump_ALL_cagr": (oth.get(("2w_even", "iv1.10_sk10_earnjump")) or {}).get("ALL_cagr"),
                             "phase_odd_ALL_cagr": (oth.get(("2w_odd", CENTRAL)) or {}).get("ALL_cagr"), "phase_odd_PRE_sharpe": (oth.get(("2w_odd", CENTRAL)) or {}).get("PRE_sharpe"),
                             "tenor4w_ALL_cagr": (oth.get(("4w_LONGER_TENOR_COMPARATOR", CENTRAL)) or {}).get("ALL_cagr"),
                             "tenor4w_PRE_sharpe": (oth.get(("4w_LONGER_TENOR_COMPARATOR", CENTRAL)) or {}).get("PRE_sharpe")})
    summ.sort(key=lambda r: -r["PRE_min_sharpe"]); _w(rd / f"{BASE}_summary.csv", summ)
    # condor / premium-selling mechanical check: central-cost ALL CAGR vs assumed IVRP
    ck = []
    for s in ("iron_condor", "bull_put_credit", "bear_call_credit", "cash_secured_put", "covered_call", "long_straddle"):
        for z in sizing:
            for cm in cms:
                row = {"structure": s, "exit": "expiry", "sizing": z, "cost_mult": cm}
                for sc in scens:
                    r = next((r for r in reg if r.get("structure") == s and r.get("exit") == "expiry" and r.get("sizing") == z and r.get("tenor") == "2w_even" and r["scenario"] == sc.name and r["cost_mult"] == cm), None)
                    if r: row[f"ALL_cagr_{sc.name}"] = r["ALL_cagr"]; row[f"PRE_sharpe_{sc.name}"] = r["PRE_sharpe"]
                vals = [row.get(f"ALL_cagr_{sc.name}") for sc in scens]
                row["monotonic_in_ivrp"] = all(a is not None and b is not None and (b >= a if s != "long_straddle" else b <= a) for a, b in zip(vals, vals[1:]))
                ck.append(row)
    _w(rd / f"{BASE}_ivrp-dependence.csv", ck)
    # nested rolling-origin: choose among option variants (2w_even) + shares canonical phase, worst-case over IV x cost
    combos = sorted({(k[3], k[4]) for k in curves if k[5] == "2w_even" and k[3] in grid})
    cand = {(s, x, z): {c: curves[(s, x, z, c[0], c[1], "2w_even")] for c in combos} for s in structs for x in exits for z in sizing}
    cand[("shares_momentum", "wd0", "p0")] = {c: sh_curves[(0, 0, c[1])] for c in combos}
    nest_rows = []; stitched = {c: np.full(len(dates), np.nan) for c in combos}
    for Y in range(2015, int(SB.END[:4]) + 1):
        tr = SB.WinIdx(dates, "2012-01-01", f"{Y-1}-12-31"); te = SB.WinIdx(dates, f"{Y}-01-01", min(f"{Y}-12-31", SB.END))
        best, bs = None, -1e9
        for k, cc in cand.items():
            v = [SB.Metrics(e, *tr)["sharpe"] for e in cc.values()]; v = min(x if x == x else -9.0 for x in v)
            if v > bs: best, bs = k, v
        if bs <= 0: best = ("cash", "", "")
        for c in combos:
            if best[0] == "cash": stitched[c][te[0]:te[1] + 1] = SB.CASH_YIELD / 252
            else:
                e = cand[best][c].astype(float); stitched[c][te[0]:te[1] + 1] = e[te[0]:te[1] + 1] / e[te[0] - 1:te[1]] - 1
        nest_rows.append({"test_year": Y, "chosen": "|".join(best), "train_worst_sharpe": bs})
    si = SB.WinIdx(dates, "2015-01-01", SB.END)[0]; nest_eq = {}
    for c, rr in stitched.items():
        e = np.full(len(dates), np.nan); e[si - 1] = ACCOUNT
        for t in range(si, len(dates)): e[t] = e[t - 1] * (1 + np.nan_to_num(rr[t]))
        nest_eq[c] = e
    nm = {c: SB.Metrics(e, si, len(dates) - 1) for c, e in nest_eq.items()}
    for c in combos: nest_rows.append({"test_year": "2015-2026_stitched", "chosen": f"combo {c[0]} cost{c[1]:g}x", **{k: nm[c][k] for k in ("cagr", "sharpe", "maxdd")}})
    _w(rd / f"{BASE}_nested-selection.csv", nest_rows)
    # equity (tidy) for top-2 options variants, shares canonical, SPY, cash, nested central
    top = [r for r in summ][:2]; cenk = (CENTRAL, CEN_COST)
    series = {f"{t['variant']}|central": curves[(t["structure"], t["exit"], t["sizing"], CENTRAL, CEN_COST, "2w_even")] for t in top}
    series["shares_momentum|wd0|p0|cost2x"] = sh_curves[(0, 0, 2.0)]; series["SPY_buyhold_totalreturn"] = spy_eq; series["cash_2pct"] = cash_eq
    series["nested_oos|central"] = nest_eq.get(cenk, next(iter(nest_eq.values())))
    eq_rows = []
    for k, e in series.items():
        e = np.asarray(e, float); pk = np.fmax.accumulate(np.where(np.isfinite(e), e, -np.inf))
        for t in range(s0 - 1, len(dates)):
            if np.isfinite(e[t]): eq_rows.append({"date": dates[t], "series": k, "equity": round(float(e[t]), 2), "drawdown_pct": round(float(e[t] / pk[t] - 1), 5)})
    _w(rd / f"{BASE}_equity.csv", eq_rows)
    trades = []
    for t in top:
        rr = RunOptions2(Cfg2(t["structure"], t["exit"], t["sizing"], cen, CEN_COST, record=True)); trades += [{**x, "scenario": f"{CENTRAL}_cost2x"} for x in rr["trades"]]
    _w(rd / f"{BASE}_trades.csv", trades)
    from Options.Report import Bootstrap
    mc = []
    for k, e in series.items():
        if k.startswith("cash") or k.startswith("nested"): continue
        e = np.asarray(e, float)[s0 - 1:]; rets = e[1:] / e[:-1] - 1
        for yrs in (1, 3, 5, 10):
            mc.append({"strategy": k, "start": ACCOUNT, "monthly_deposit": 0, "years": yrs, "ILLUSTRATIVE_ONLY": True, **Bootstrap(rets, ACCOUNT, yrs, 0)})
    _w(rd / f"{BASE}_montecarlo.csv", mc)
    meta = {"base_name": BASE, "run_id": run_id, "run_at_et": run_at.isoformat(timespec="seconds"), "code_version": CODE_VERSION, "end": SB.END,
            "simulated_vs_real": "SIMULATED model option prices on real underlying bars; real 2026 option panel used only for sanity/IVRP range",
            "n_option_runs": len(jobs), "n_variants_core": len(structs) * len(exits) * len(sizing), "shares_phase_runs": len(sh), "scenarios": [asdict(s) for s in scens + STRESS],
            "central": [CENTRAL, CEN_COST], "fill_model": FILL_MODEL, "itm_frac": ITM_FRAC, "sizing": SIZING, "ticks": {**TICKS, "default": TICK_DEFAULT}, "tenors": TENORS, "account": ACCOUNT, "deposits": 0,
            "spy_vol_ALL": spy_vol, "top2_by_PRE_min_sharpe": [t["variant"] for t in top], "nested": nm.get(cenk), "elapsed_s": time.time() - t0,
            "contamination": "IVRP grid {1.0,1.1,1.25} chosen after seeing the 2026 real panel and v1 results; every window incl. HO has been viewed in v1 -> nothing is untouched"}
    (rd / f"{BASE}_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({"top": [(t["variant"], round(t["PRE_min_sharpe"], 3), round(t["central_ALL_cagr"], 4)) for t in summ[:8]], "nested": nm.get(cenk)}, default=str, indent=1))
    print("DONE", rd, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
    import Options.Validate10k as _M; _M.main()
