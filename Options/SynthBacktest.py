"""Synthetic-options backtest engine (MODEL prices, not historical quotes).

Daily OHLC only: signal from previous close, entries/exits at next OPEN, positions held overnight
(main design creates 0 day trades; DayTradeGuard enforces max 3 per rolling 5 sessions anyway).
Biweekly cycle: enter at the open of the first session of even weeks, synthetic expiry = last session of the
following week (~9-10 sessions, Tim's rolling ~2-week preference). Whole contracts (x100), buy at model ask /
sell at model bid, per-contract fees, cash accounting with collateral reserves, cash yield assumed.

Usage:
  .venv\\Scripts\\python -m Options.SynthBacktest --procs 20                (full phase-1 grid)
  .venv\\Scripts\\python -m Options.SynthBacktest --quick                  (smoke)
"""
from __future__ import annotations
import argparse, csv, json, math, time
from dataclasses import dataclass, asdict
from datetime import date, datetime
from multiprocessing import Pool
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np

from Options.Pricing import BsPrice, BsGreeks, Intrinsic, PayoffExtremes
from Options.VolModel import TrailingVol, IvScenario, ImpliedVol
from Options.Structures import STRUCTURES, REJECTED, BuildLegs
from Options.DayTradeGuard import DayTradeGuard

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data" / "daily"
EARN = ROOT / "data" / "earnings"
OUT = ROOT / "out"
CODE_VERSION = "stonk-data-0.4.0-synthopt"

TICKERS = ["SPY", "QQQ", "IWM", "XBI", "GDX", "TLT", "HOOD", "AXTI", "SHOP", "QURE"]
EVENT_NAMES = {"HOOD", "AXTI", "SHOP", "QURE"}
# ASSUMED half-spread as fraction of option mid (cost_mult scales it), and minimum half-spread in $
# Calibrated (contemporary, 2026-11-09 ~15:29 ET Robinhood near-ATM Nov-20 quotes): SPY ~0.3-0.5%, HOOD ~1-2%,
# SHOP ~2-8%, AXTI ~5-10%, QURE ~10-30%. XBI/GDX/TLT not sampled -> assumed 2%. Historical spreads (esp. 2012-15)
# were likely wider: use cost_mult 2x/3x as the stress.
HALF_SPREAD_PCT = {"SPY": 0.005, "QQQ": 0.005, "IWM": 0.008, "XBI": 0.02, "GDX": 0.02, "TLT": 0.02,
                   "HOOD": 0.015, "SHOP": 0.03, "AXTI": 0.08, "QURE": 0.15}
MIN_HALF = {"SPY": 0.005, "QQQ": 0.005, "IWM": 0.005}
MIN_HALF_DEFAULT = 0.025
FEE_PER_CONTRACT = 0.03      # per leg per side, $ (regulatory-fee estimate; $0 commission assumed)
STOCK_SLIP_BPS = 5.0
RATE = 0.02                  # flat risk-free assumption for pricing
CASH_YIELD = 0.02            # flat assumed yield on cash (incl. collateral)
RISK_FRAC = 0.10             # default (r10) max loss per new position as fraction of equity
PORT_RISK_CAP = 0.50         # default (r10) cap on sum of open max losses
RISK_LEVELS = {"r02": (0.02, 0.10), "r05": (0.05, 0.25), "r10": (0.10, 0.50)}   # sizing is part of each variant
TICKER_CAP = 0.50            # Tim's 50% per ticker/position cap (on max loss / collateral)
DEFAULT_JUMP = 0.08
SIM_START, END = "2012-01-03", "2026-11-06"
WINDOWS = {"EARLY": ("2012-01-01", "2017-12-31"), "IS": ("2018-01-01", "2025-01-30"),
           "HO": ("2025-01-31", END), "PRE": ("2012-01-01", "2025-01-30"), "ALL": ("2012-01-01", END)}
EXITS = ["expiry", "tminus1", "pt_sl"]
_D = {}


def LoadData(end=END):
    meta = {}
    spy = json.loads((DAILY / "SPY.json").read_text())
    dates = [b["date"] for b in spy["bars"] if "2011-01-01" <= b["date"] <= end]
    di = {d: i for i, d in enumerate(dates)}
    T = len(dates)
    tick = {}
    for s in TICKERS:
        p = DAILY / f"{s}.json"
        if not p.exists():
            continue
        js = json.loads(p.read_text())
        O = np.full(T, np.nan); C = np.full(T, np.nan)
        for b in js["bars"]:
            i = di.get(b["date"])
            if i is not None and b["open"] > 0 and b["close"] > 0:
                O[i], C[i] = b["open"], b["close"]
        Cff = C.copy()
        for i in range(1, T):
            if not np.isfinite(Cff[i]): Cff[i] = Cff[i - 1]
        vol = TrailingVol(Cff)
        sma50 = np.array([np.nanmean(C[max(0, i - 49):i + 1]) if i >= 49 else np.nan for i in range(T)])
        r63 = np.full(T, np.nan); r63[63:] = Cff[63:] / Cff[:-63] - 1
        bull = (C > sma50) & (r63 > 0); bear = (C < sma50) & (r63 < 0)
        eff, cov_first = [], None
        ep = EARN / f"{s}.json"
        if ep.exists():
            from DataPull.market_calendar import next_session
            ej = json.loads(ep.read_text())
            cov_first = ej.get("coverage_first")
            for r in ej.get("dates", []):
                d = date.fromisoformat(r["date"]); e = d if r["timing"] == "BMO" else next_session(d)
                if e.isoformat() in di: eff.append(di[e.isoformat()])
        tick[s] = {"O": O, "C": C, "Cff": Cff, "base": vol["base"], "lr": vol["lr"], "bull": bull, "bear": bear,
                   "eff": np.array(sorted(set(eff)), dtype=int), "earn_cov_first": cov_first}
        meta[s] = {"source": js.get("source"), "fetched_at_et": js.get("fetched_at_et"), "adjustment_type": js.get("adjustment_type"),
                   "first": js["bars"][0]["date"], "last": js["bars"][-1]["date"], "earnings_coverage_first": cov_first,
                   "n_earnings_in_range": len(eff)}
    wo = np.array([(date.fromisoformat(d) - date(2000, 1, 3)).days // 7 for d in dates])
    entries = {}
    for t in range(1, T):
        if wo[t] != wo[t - 1] and wo[t] % 2 == 0:
            nxt = np.where(wo == wo[t] + 1)[0]
            if len(nxt):
                entries[t] = int(nxt[-1])
    return {"dates": dates, "tick": tick, "entries": entries, "meta": meta}


def Tyears(dates, t, exp, at_open):
    days = (date.fromisoformat(dates[exp]) - date.fromisoformat(dates[t])).days
    return (days + (6.5 / 24 if at_open else 0.0)) / 365.0


def HalfSpread(sym, mid, cost_mult):
    return cost_mult * max(MIN_HALF.get(sym, MIN_HALF_DEFAULT), HALF_SPREAD_PCT[sym] * mid)


def EventVar(tk, t, exp, scen):
    if scen.earn_mode != "price_jump" or len(tk["eff"]) == 0:
        return 0.0
    k = np.searchsorted(tk["eff"], t, side="right")
    if k < len(tk["eff"]) and tk["eff"][k] <= exp:
        past = tk["eff"][tk["eff"] <= t]
        moves = [abs(tk["lr"][e]) for e in past if np.isfinite(tk["lr"][e])]
        j = float(np.mean(moves)) if moves else DEFAULT_JUMP
        return j * j
    return 0.0


def LegMids(sym, tk, legs, S, base, T, scen, event_var, at_entry):
    out = []
    for lg in legs:
        if lg["kind"] == "S":
            out.append(S); continue
        iv = ImpliedVol(base, S, lg["K"], T, scen, event_var, at_entry, sym in EVENT_NAMES)
        out.append(BsPrice(S, lg["K"], T, RATE, 0.0, iv, lg["kind"]))
    return out


def PackageFill(sym, legs, mids, S, cost_mult, opening):
    """$ cash flow per package (positive = we pay). opening=True: buy +qty legs at ask, sell -qty at bid.
    Closing reverses each leg (sell longs at bid, buy back shorts at ask)."""
    flow = 0.0; fees = 0.0
    for lg, m in zip(legs, mids):
        q = lg["qty"] if opening else -lg["qty"]
        if lg["kind"] == "S":
            px = m * (1 + cost_mult * STOCK_SLIP_BPS / 1e4 * (1 if q > 0 else -1))
            flow += q * px * 100
        else:
            hs = HalfSpread(sym, m, cost_mult)
            px = m + hs if q > 0 else max(m - hs, 0.0)
            flow += q * px * 100
            fees += abs(q) * FEE_PER_CONTRACT * cost_mult
    return flow + fees


def PackageMid(legs, mids, T, mn, mx):
    v = sum(lg["qty"] * m for lg, m in zip(legs, mids))
    lo = mn * math.exp(-RATE * max(T, 0)) if math.isfinite(mn) else -math.inf
    return min(max(v, lo), mx)


@dataclass
class Cfg:
    structure: str
    exit: str
    scen: IvScenario
    cost_mult: float
    account: float
    record: bool = False


def _init(end):
    import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
    _D.update(LoadData(end))


def RunOptions(cfg: Cfg, D=None):
    D = D or _D
    dates, tick, entries = D["dates"], D["tick"], D["entries"]
    T = len(dates)
    s0 = next(i for i, d in enumerate(dates) if d >= SIM_START)
    bias = STRUCTURES[cfg.structure][0]
    rule, _, rk = cfg.exit.partition("|")
    risk_frac, port_cap = RISK_LEVELS.get(rk or "r10")
    scen = cfg.scen
    cash = cfg.account
    eq = np.full(T, np.nan)
    positions = []; trades = []; greeks = []
    guard = DayTradeGuard(dates)
    stats = {"entries": 0, "infeasible": 0, "earn_skip": 0, "earn_unknown_entries": 0, "unlimited_reject": 0, "bad_pricing": 0}
    pid = 0

    def close_pos(p, t, proceeds_pkg, how):
        nonlocal cash
        cash += p["n"] * proceeds_pkg
        pnl = p["n"] * (proceeds_pkg - p["entry_cost_pkg"])
        for k, lg in enumerate(p["legs"]):
            guard.RecordClose(dates[t], f"{p['id']}:{k}", p["id"], p["n"] * abs(lg["qty"]))
        trades.append({"entry_date": dates[p["t0"]], "exit_date": dates[t], "symbol": p["sym"], "side": cfg.structure,
                       "qty": p["n"], "entry_price": round(p["entry_cost_pkg"] / 100, 4), "exit_price": round(proceeds_pkg / 100, 4),
                       "pnl": round(pnl, 2), "exit_how": how, "max_loss": round(p["max_loss"], 2),
                       "legs": ";".join(f"{lg['qty']:+d}{lg['kind']}{lg['K']:g}" for lg in p["legs"]), "expiry": dates[p["exp"]]})

    for t in range(s0, T):
        # ---- exits at open ----
        for p in list(positions):
            if p["exit_at"] == t and p["exit_at"] >= 0:
                tk = tick[p["sym"]]
                S = tk["O"][t] if np.isfinite(tk["O"][t]) else tk["Cff"][t - 1]
                Ty = Tyears(dates, t, p["exp"], True)
                mids = LegMids(p["sym"], tk, p["legs"], S, tk["base"][t - 1], Ty, scen, EventVar(tk, t, p["exp"], scen), False)
                proceeds = -PackageFill(p["sym"], p["legs"], mids, S, cfg.cost_mult, opening=False)
                assert guard.CanClose(dates[t], [(f"{p['id']}:{k}", 1) for k in range(len(p["legs"]))])
                close_pos(p, t, proceeds, "open_exit"); positions.remove(p)
        # ---- entries at open ----
        if t in entries:
            exp = entries[t]
            E = eq[t - 1] if np.isfinite(eq[t - 1]) else cash
            held = {p["sym"] for p in positions}
            for sym in TICKERS:
                if sym not in tick or sym in held:
                    continue
                tk = tick[sym]
                if not (np.isfinite(tk["O"][t]) and np.isfinite(tk["C"][t - 1]) and np.isfinite(tk["base"][t - 1])):
                    continue
                if bias == "bull" and not tk["bull"][t - 1]: continue
                if bias == "bear" and not tk["bear"][t - 1]: continue
                exit_at = -1 if rule == "expiry" else exp - 1   # -1: settle at expiry close
                if rule != "expiry" and exit_at <= t:
                    continue
                spans_earn = len(tk["eff"]) > 0 and np.any((tk["eff"] > t - 1) & (tk["eff"] <= exp))
                if sym in EVENT_NAMES:
                    cov = tk["earn_cov_first"]
                    if cov is None or dates[t] < cov:
                        stats["earn_unknown_entries"] += 1
                if spans_earn and scen.earn_mode == "skip":
                    stats["earn_skip"] += 1; continue
                S = tk["O"][t]; Ty = Tyears(dates, t, exp, True)
                ev = EventVar(tk, t, exp, scen)
                atm = ImpliedVol(tk["base"][t - 1], S, S, Ty, scen, 0.0, True, sym in EVENT_NAMES)
                legs = BuildLegs(cfg.structure, S, atm, Ty)
                mn, mx = PayoffExtremes(legs)
                if not math.isfinite(mn):
                    stats["unlimited_reject"] += 1; continue
                mids = LegMids(sym, tk, legs, S, tk["base"][t - 1], Ty, scen, ev, True)
                cost_pkg = PackageFill(sym, legs, mids, S, cfg.cost_mult, opening=True)
                max_loss = cost_pkg - mn * 100
                if max_loss <= 0 or (math.isfinite(mx) and cost_pkg > mx * 100):
                    stats["bad_pricing"] += 1; continue
                free = cash - sum(p["reserve"] for p in positions)
                open_risk = sum(p["max_loss"] for p in positions)
                budget = min(risk_frac * E, TICKER_CAP * E, free, port_cap * E - open_risk)
                n = int(budget // max_loss)
                if n < 1:
                    stats["infeasible"] += 1; continue
                pid += 1
                cash -= n * cost_pkg
                p = {"id": pid, "sym": sym, "legs": legs, "n": n, "t0": t, "exp": exp, "exit_at": exit_at,
                     "entry_cost_pkg": cost_pkg, "max_loss": n * max_loss, "reserve": n * (max_loss - cost_pkg), "mn": mn, "mx": mx}
                positions.append(p); stats["entries"] += 1
                for k, lg in enumerate(legs):
                    guard.RecordOpen(dates[t], f"{pid}:{k}", pid, n * abs(lg["qty"]))
                if cfg.record:
                    g = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
                    for lg in legs:
                        if lg["kind"] == "S":
                            g["delta"] += lg["qty"] * 100 * n; continue
                        iv = ImpliedVol(tk["base"][t - 1], S, lg["K"], Ty, scen, ev, True, sym in EVENT_NAMES)
                        gg = BsGreeks(S, lg["K"], Ty, RATE, 0.0, iv, lg["kind"])
                        for kk in g: g[kk] += lg["qty"] * 100 * n * gg[kk]
                    greeks.append({"date": dates[t], "symbol": sym, "structure": cfg.structure, "contracts": n, "spot": round(S, 4),
                                   "atm_iv": round(atm, 4), "T_days": round(Ty * 365, 2), "pkg_cost": round(cost_pkg, 2),
                                   "max_loss_pkg": round(max_loss, 2), **{k: round(v, 4) for k, v in g.items()}})
        # ---- close: settle / mark / pt-sl ----
        mark = 0.0
        for p in list(positions):
            tk = tick[p["sym"]]
            S = tk["Cff"][t]
            if rule == "expiry" and t == p["exp"]:
                proceeds = sum(lg["qty"] * (S if lg["kind"] == "S" else Intrinsic(S, lg["K"], lg["kind"])) for lg in p["legs"]) * 100
                close_pos(p, t, proceeds, "expiry_settle"); positions.remove(p); continue
            Ty = Tyears(dates, t, p["exp"], False)
            mids = LegMids(p["sym"], tk, p["legs"], S, tk["base"][t], Ty, scen, EventVar(tk, t, p["exp"], scen), False)
            v = p["n"] * 100 * PackageMid(p["legs"], mids, Ty, p["mn"], p["mx"])
            mark += v
            if rule == "pt_sl" and p["exit_at"] > t + 1:  # signal at close t -> fill next open (gap risk realised)
                pnl = v - p["n"] * p["entry_cost_pkg"]
                if pnl >= 0.5 * p["max_loss"] or pnl <= -0.5 * p["max_loss"]:
                    p["exit_at"] = t + 1
        cash += cash * CASH_YIELD / 252
        eq[t] = cash + mark
    res = {"eq": eq.astype(np.float32), "stats": stats, "n_trades": len(trades),
           "day_trades_leg": len(guard.day_trades), "day_trades_pkg": len({(x[0], x[2]) for x in guard.day_trades}),
           "win_rate": float(np.mean([x["pnl"] > 0 for x in trades])) if trades else float("nan")}
    if cfg.record:
        res["trades"] = trades; res["greeks"] = greeks
    return res


def RunShares(cost_mult, account, D=None):
    """Baseline: equal-weight fractional shares of bull-signal tickers, 50% cap, same biweekly entry days."""
    D = D or _D
    dates, tick, entries = D["dates"], D["tick"], D["entries"]
    T = len(dates); s0 = next(i for i, d in enumerate(dates) if d >= SIM_START)
    cash = account; qty = {}; eq = np.full(T, np.nan)
    for t in range(s0, T):
        if t in entries:
            E = cash + sum(q * tick[s]["Cff"][t - 1] for s, q in qty.items())
            picks = [s for s in TICKERS if s in tick and tick[s]["bull"][t - 1] and np.isfinite(tick[s]["O"][t])]
            w = min(0.5, 1.0 / len(picks)) if picks else 0.0
            target = {s: w * E / tick[s]["O"][t] for s in picks}
            for s in set(qty) | set(target):
                o = tick[s]["O"][t] if np.isfinite(tick[s]["O"][t]) else tick[s]["Cff"][t - 1]
                dq = target.get(s, 0.0) - qty.get(s, 0.0)
                cash -= dq * o + abs(dq) * o * cost_mult * STOCK_SLIP_BPS / 1e4
                qty[s] = target.get(s, 0.0)
            qty = {s: q for s, q in qty.items() if q > 0}
        cash += cash * CASH_YIELD / 252 if cash > 0 else 0.0
        eq[t] = cash + sum(q * tick[s]["Cff"][t] for s, q in qty.items())
    return {"eq": eq.astype(np.float32)}


def RunCash(account, D=None):
    D = D or _D
    dates = D["dates"]; T = len(dates); s0 = next(i for i, d in enumerate(dates) if d >= SIM_START)
    eq = np.full(T, np.nan); v = account
    for t in range(s0, T):
        v *= 1 + CASH_YIELD / 252; eq[t] = v
    return {"eq": eq.astype(np.float32)}


def WinIdx(dates, a, b):
    s = next(i for i, d in enumerate(dates) if d >= a); e = max(i for i, d in enumerate(dates) if d <= b)
    return s, e


def Metrics(eq, s, e):
    seg = eq[s - 1:e + 1].astype(float) if s > 0 and np.isfinite(eq[s - 1]) else eq[s:e + 1].astype(float)
    seg = seg[np.isfinite(seg)]
    if len(seg) < 30 or seg[0] <= 0:
        return {"cagr": float("nan"), "sharpe": float("nan"), "maxdd": float("nan"), "ruin": False}
    r = seg[1:] / np.where(seg[:-1] > 0, seg[:-1], np.nan) - 1
    r = r[np.isfinite(r)]
    tot = seg[-1] / seg[0]
    ex = r - CASH_YIELD / 252                       # Sharpe on returns in excess of the assumed cash yield
    sd = ex.std(ddof=1)
    pk = np.maximum.accumulate(seg)
    vol = sd * math.sqrt(252)
    return {"cagr": float(tot ** (252 / len(r)) - 1) if tot > 0 else -1.0,
            "sharpe": float(ex.mean() / sd * math.sqrt(252)) if vol > 0.005 else float("nan"),   # nan = ~no risk taken (idle/dead)
            "vol": float(vol), "maxdd": float((seg / pk - 1).min()), "ruin": bool(seg.min() < 0.1 * seg[0])}


SCENARIOS = [IvScenario(f"iv{m:.1f}_sk{int(sk*100)}", ivrp=m, skew=sk) for m in (1.0, 1.2, 1.4) for sk in (0.0, 0.10)]
STRESS = [IvScenario("iv1.2_sk10_crush30", 1.2, 0.10, crush=0.30), IvScenario("iv1.2_sk10_earnjump", 1.2, 0.10, earn_mode="price_jump")]
CENTRAL = "iv1.2_sk10"


def _job(args):
    structure, exit_, scen, cm, acct = args
    r = RunOptions(Cfg(structure, exit_, scen, cm, acct))
    return args, r


def main():
    from Options.Report import WriteOutputs
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=20)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--slug", default="synthetic-options")
    args = ap.parse_args()
    t0 = time.time(); run_at = datetime.now(ET)
    run_id = f"so_{run_at.strftime('%Y%m%dT%H%M%S')}" + ("_quick" if args.quick else "")
    run_dir = OUT / run_id; run_dir.mkdir(parents=True, exist_ok=False)
    accounts = [250.0, 1000.0, 10000.0]
    structs = list(STRUCTURES); scens = SCENARIOS; cms = [1.0, 2.0, 3.0]; exits = EXITS; stress = STRESS
    if args.quick:
        structs = ["bull_call_debit", "iron_condor", "long_straddle"]; scens = SCENARIOS[2:4]; cms = [1.0]; exits = ["tminus1"]; accounts = [1000.0, 10000.0]; stress = []
    exits = [f"{e}|{rk}" for e in exits for rk in RISK_LEVELS]
    jobs = [(s, x, sc, cm, a) for s in structs for x in exits for sc in scens for cm in cms for a in accounts]
    jobs += [(s, x, sc, 2.0, a) for s in structs for x in exits for sc in stress for a in accounts]
    print(f"{run_id}: {len(jobs)} runs, {len(structs)*len(exits)} strategy variants", flush=True)
    _init(END)
    with Pool(args.procs, initializer=_init, initargs=(END,)) as pool:
        res = pool.map(_job, jobs, chunksize=2)
    print(f"grid done {time.time()-t0:.0f}s", flush=True)
    base = {("shares_momentum", cm, a): RunShares(cm, a) for cm in cms for a in accounts}
    cash = {a: RunCash(a) for a in accounts}
    WriteOutputs(run_dir, run_id, run_at, res, base, cash, structs, exits, accounts, args, t0)


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)
    import Options.SynthBacktest as _M   # run via the importable module so workers/Report share state
    _M.main()
