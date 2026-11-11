"""Outputs for a synthetic-options run: registry, robustness summary, nested rolling-origin selection,
equity/trade/Greeks CSVs (tidy, Graphtard-ready), block-bootstrap wealth scenarios, meta."""
from __future__ import annotations
import csv, json, math, time
from dataclasses import asdict
import numpy as np

from Options import SynthBacktest as SB

BASE = "2026-11-09_synthetic-options"


def _w(path, rows, cols):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})


def Dd(eq):
    e = eq.astype(float); pk = np.fmax.accumulate(np.where(np.isfinite(e), e, -np.inf)); return e / pk - 1


def Bootstrap(rets, start, years, dep, paths=2000, block=21, seed=7):
    rng = np.random.default_rng(seed)
    r = rets[np.isfinite(rets)]
    H = int(252 * years); nb = math.ceil(H / block)
    st = rng.integers(0, len(r) - block, size=(paths, nb))
    R = r[(st[:, :, None] + np.arange(block)).reshape(paths, -1)[:, :H]]
    bal = np.full(paths, start, float); pk = bal.copy(); mdd = np.zeros(paths); contrib = np.full(paths, start, float)
    for d in range(H):
        bal = np.maximum(bal * (1 + R[:, d]), 0.0)
        if dep and d % 21 == 20:
            bal += dep; contrib += dep
        pk = np.maximum(pk, bal); mdd = np.minimum(mdd, np.where(pk > 0, bal / pk - 1, 0))
    return {"median_terminal": float(np.median(bal)), "p10_terminal": float(np.percentile(bal, 10)), "p90_terminal": float(np.percentile(bal, 90)),
            "contributions": float(contrib[0]), "p_severe_loss_lt50pct_contrib": float(np.mean(bal < 0.5 * contrib)),
            "p_dd_worse_80pct": float(np.mean(mdd <= -0.8)), "median_maxdd": float(np.median(mdd)), "p_1m": float(np.mean(bal >= 1e6))}


def WriteOutputs(run_dir, run_id, run_at, res, base, cash, structs, exits, accounts, args, t0):
    D = SB._D; dates = D["dates"]
    W = {k: SB.WinIdx(dates, *v) for k, v in SB.WINDOWS.items()}
    variants = [(s, x) for s in structs for x in exits]
    n_var = len(variants); hc = math.sqrt(math.log(max(n_var, 2)))
    reg, curves = [], {}
    for (s, x, sc, cm, a), r in res:
        m = {w: SB.Metrics(r["eq"], *W[w]) for w in W}
        row = {"structure": s, "exit": x, "variant": f"{s}|{x}", "scenario": sc.name, "ivrp": sc.ivrp, "skew": sc.skew, "crush": sc.crush,
               "earn_mode": sc.earn_mode, "cost_mult": cm, "account": a, "n_trades": r["n_trades"], "win_rate": r["win_rate"],
               "day_trades_leg": r["day_trades_leg"], "day_trades_pkg": r["day_trades_pkg"], **r["stats"],
               "final_equity": float(r["eq"][-1])}
        for w in W:
            for k in ("cagr", "sharpe", "maxdd"):
                row[f"{w}_{k}"] = m[w][k]
        row["ruin_ALL"] = m["ALL"]["ruin"]
        reg.append(row); curves[(s, x, sc.name, cm, a)] = r["eq"]
    for (nm, cm, a), r in base.items():
        m = {w: SB.Metrics(r["eq"], *W[w]) for w in W}
        row = {"structure": nm, "exit": "biweekly_rebalance", "variant": nm, "scenario": "n/a", "cost_mult": cm, "account": a}
        for w in W:
            for k in ("cagr", "sharpe", "maxdd"): row[f"{w}_{k}"] = m[w][k]
        reg.append(row)
    cols = list(reg[0].keys())
    for r in reg:
        for k in r:
            if k not in cols: cols.append(k)
    _w(run_dir / f"{BASE}_registry.csv", reg, cols)

    grid_scen = [sc.name for sc in SB.SCENARIOS]
    summ = []
    for a in accounts:
        for s, x in variants:
            rows = [r for r in reg if r.get("structure") == s and r.get("exit") == x and r.get("account") == a]
            core = [r for r in rows if r["scenario"] in grid_scen]
            if not core: continue
            cen = next((r for r in core if r["scenario"] == SB.CENTRAL and r["cost_mult"] == 2.0), core[0])
            pre = [r["PRE_sharpe"] for r in core]
            stress = {r["scenario"]: r["ALL_cagr"] for r in rows if r["scenario"] not in grid_scen}
            tot_try = cen["entries"] + cen["infeasible"]
            summ.append({"account": a, "variant": f"{s}|{x}", "structure": s, "exit": x,
                         "PRE_min_sharpe": float(min(v if v == v else -9.0 for v in pre)), "PRE_median_sharpe": float(np.nanmedian(pre)) if any(v == v for v in pre) else float("nan"),
                         "PRE_nan_sharpe_runs": int(sum(1 for v in pre if v != v)),
                         "PRE_frac_cagr_pos": float(np.mean([r["PRE_cagr"] > 0 for r in core])),
                         "ALL_worst_maxdd": float(np.nanmin([r["ALL_maxdd"] for r in core])),
                         "any_ruin": bool(any(r["ruin_ALL"] for r in core)),
                         **{f"central_{w}_{k}": cen[f"{w}_{k}"] for w in ("EARLY", "IS", "HO", "ALL") for k in ("cagr", "sharpe", "maxdd")},
                         "central_ALL_haircut_sharpe": cen["ALL_sharpe"] / hc, "central_PRE_sharpe": cen["PRE_sharpe"], "central_PRE_cagr": cen["PRE_cagr"],
                         "central_trades": cen["n_trades"], "central_infeasible_frac": cen["infeasible"] / tot_try if tot_try else float("nan"),
                         "central_earn_unknown_entries": cen["earn_unknown_entries"],
                         "stress_crush30_ALL_cagr": stress.get("iv1.2_sk10_crush30"), "stress_earnjump_ALL_cagr": stress.get("iv1.2_sk10_earnjump")})
    for a in accounts:
        for cm in (1.0, 2.0, 3.0):
            r = next((r for r in reg if r.get("variant") == "shares_momentum" and r["cost_mult"] == cm and r["account"] == a), None)
            if r and cm == 2.0 or (r and len({x["cost_mult"] for x in reg if x.get("variant") == "shares_momentum"}) == 1):
                summ.append({"account": a, "variant": "shares_momentum", "structure": "shares_momentum", "exit": "biweekly",
                             "PRE_min_sharpe": min(x["PRE_sharpe"] for x in reg if x.get("variant") == "shares_momentum" and x["account"] == a),
                             **{f"central_{w}_{k}": r[f"{w}_{k}"] for w in ("EARLY", "IS", "HO", "ALL") for k in ("cagr", "sharpe", "maxdd")}})
    summ.sort(key=lambda r: (r["account"], -(r["PRE_min_sharpe"] if r["PRE_min_sharpe"] == r["PRE_min_sharpe"] else -9)))
    _w(run_dir / f"{BASE}_summary.csv", summ, list(dict.fromkeys(k for r in summ for k in r)))

    # ---- nested rolling-origin selection (by worst-case train Sharpe across the 18 IV x cost runs) ----
    combos = sorted({(k[2], k[3]) for k in curves if k[2] in grid_scen})
    CEN = (SB.CENTRAL, 2.0) if (SB.CENTRAL, 2.0) in combos else next((c for c in combos if c[0] == SB.CENTRAL), combos[0])
    cms_run = sorted({c[1] for c in combos}); CM_STRESS = max(cms_run)
    nested = {}; nested_rows = []
    for a in accounts:
        stitched = {c: np.full(len(dates), np.nan) for c in combos}
        y0 = 2015
        for Y in range(y0, int(SB.END[:4]) + 1):
            tr = SB.WinIdx(dates, "2012-01-01", f"{Y-1}-12-31"); te = SB.WinIdx(dates, f"{Y}-01-01", min(f"{Y}-12-31", SB.END))
            best, bscore = None, -1e9
            for s, x in variants:
                vals = [SB.Metrics(curves[(s, x, sc, cm, a)], *tr)["sharpe"] for sc, cm in combos if (s, x, sc, cm, a) in curves]
                if not vals: continue
                vals = [v if v == v else -9.0 for v in vals]   # nan (no risk/dead) counts as worst
                sc_ = float(min(vals))
                if sc_ > bscore: best, bscore = (s, x), sc_
            if bscore <= 0:
                best = ("cash", "")  # no variant with positive worst-case train Sharpe -> stay in cash
            for c in combos:
                if best[0] == "cash":
                    stitched[c][te[0]:te[1] + 1] = SB.CASH_YIELD / 252
                else:
                    e = curves[(best[0], best[1], c[0], c[1], a)].astype(float)
                    stitched[c][te[0]:te[1] + 1] = e[te[0]:te[1] + 1] / e[te[0] - 1:te[1]] - 1
            nested_rows.append({"account": a, "test_year": Y, "chosen": "|".join(best), "train_worst_sharpe": bscore})
        s_idx = SB.WinIdx(dates, f"{y0}-01-01", SB.END)[0]
        eqs = {}
        for c, rr in stitched.items():
            e = np.full(len(dates), np.nan); e[s_idx - 1] = a
            for t in range(s_idx, len(dates)):
                e[t] = e[t - 1] * (1 + np.nan_to_num(rr[t]))
            eqs[c] = e
        mets = {c: SB.Metrics(e, s_idx, len(dates) - 1) for c, e in eqs.items()}
        worst = min(mets, key=lambda c: mets[c]["sharpe"])
        nested[a] = {"central": mets[CEN], "worst_combo": "|".join(map(str, worst)), "worst": mets[worst],
                     "ho_central": SB.Metrics(eqs[CEN], *W["HO"]), "eq_central": eqs[CEN], "start": dates[s_idx]}
    _w(run_dir / f"{BASE}_nested-selection.csv", nested_rows, ["account", "test_year", "chosen", "train_worst_sharpe"])

    # ---- top variants, detailed record ----
    top10k = [r for r in summ if r["account"] == accounts[-1] and r["variant"] != "shares_momentum"][:3]
    trades_rows, greek_rows = [], []
    for a in accounts:
        for tv in top10k[:2]:
            sc = next(s for s in SB.SCENARIOS if s.name == SB.CENTRAL)
            rr = SB.RunOptions(SB.Cfg(tv["structure"], tv["exit"], sc, CEN[1], a, record=True))
            for t in rr["trades"]: trades_rows.append({**t, "account": a, "scenario": SB.CENTRAL + "_cost2x"})
            for g in rr["greeks"]: greek_rows.append({**g, "account": a, "scenario": SB.CENTRAL + "_cost2x"})
    _w(run_dir / f"{BASE}_trades.csv", trades_rows, ["entry_date", "exit_date", "symbol", "side", "qty", "entry_price", "exit_price", "pnl",
                                                     "exit_how", "max_loss", "legs", "expiry", "account", "scenario"])
    _w(run_dir / f"{BASE}_greeks.csv", greek_rows, ["date", "symbol", "structure", "contracts", "spot", "atm_iv", "T_days", "pkg_cost",
                                                    "max_loss_pkg", "delta", "gamma", "vega", "theta", "account", "scenario"])

    # ---- equity (tidy) ----
    spy = D["tick"]["SPY"]["Cff"].astype(float)
    s0 = next(i for i, d in enumerate(dates) if d >= SB.SIM_START)
    series = {}
    for a in sorted({accounts[-1], 1000.0} & set(accounts), reverse=True):
        for tv in top10k:
            series[f"{tv['variant']}|central|${int(a)}"] = curves[(tv["structure"], tv["exit"], CEN[0], CEN[1], a)]
        series[f"shares_momentum|cost{CEN[1]:g}x|${int(a)}"] = base[("shares_momentum", CEN[1], a)]["eq"]
        series[f"nested_oos|central|${int(a)}"] = nested[a]["eq_central"]
    series[f"cash_{SB.CASH_YIELD:.0%}|${int(accounts[-1])}"] = cash[accounts[-1]]["eq"]
    series[f"SPY_buyhold|${int(accounts[-1])}"] = spy / spy[s0 - 1] * accounts[-1]
    eq_rows = []
    for nm, e in series.items():
        e = np.asarray(e, float); dd = Dd(e)
        for t in range(s0 - 1, len(dates)):
            if np.isfinite(e[t]):
                eq_rows.append({"date": dates[t], "series": nm, "equity": round(float(e[t]), 2), "drawdown_pct": round(float(dd[t]), 5)})
    _w(run_dir / f"{BASE}_equity.csv", eq_rows, ["date", "series", "equity", "drawdown_pct"])

    # ---- block bootstrap wealth scenarios ----
    mc_rows = []
    cands = [(tv["structure"], tv["exit"]) for tv in top10k[:2]]
    for a in accounts:
        for s, x in cands + [("shares_momentum", "")]:
            if s == "shares_momentum":
                cases = {"central": base[("shares_momentum", CEN[1], a)]["eq"], f"stress(cost{CM_STRESS:g}x)": base[("shares_momentum", CM_STRESS, a)]["eq"]}
            else:
                cen = curves[(s, x, CEN[0], CEN[1], a)]
                wk = min(((sc, cm) for sc, cm in combos if (s, x, sc, cm, a) in curves), key=lambda c: SB.Metrics(curves[(s, x, c[0], c[1], a)], *W["ALL"])["cagr"])
                cases = {"central": cen, f"stress({wk[0]},cost{wk[1]:g}x)": curves[(s, x, wk[0], wk[1], a)]}
            for case, e in cases.items():
                e = np.asarray(e, float)[s0 - 1:]; rets = e[1:] / e[:-1] - 1
                for yrs in (1, 3, 5, 10):
                    for dep in (0, 100, 500):
                        b = Bootstrap(rets, a, yrs, dep)
                        mc_rows.append({"strategy": f"{s}|{x}" if x else s, "case": case, "start": a, "years": yrs, "monthly_deposit": dep, **b})
    _w(run_dir / f"{BASE}_montecarlo.csv", mc_rows, list(mc_rows[0].keys()))

    meta = {"base_name": BASE, "run_id": run_id, "run_at_et": run_at.isoformat(timespec="seconds"), "code_version": SB.CODE_VERSION,
            "simulated_vs_real": "SIMULATED with MODEL option prices (no historical option quotes)", "end": SB.END,
            "tickers": SB.TICKERS, "data_meta": D["meta"], "n_strategy_variants": n_var, "n_runs": len(res),
            "haircut_divisor_sqrt_ln_N": hc, "scenarios": [asdict(s) for s in SB.SCENARIOS + SB.STRESS], "cost_mults": [1, 2, 3],
            "assumptions": {"rate": SB.RATE, "cash_yield": SB.CASH_YIELD, "risk_levels(per_position,portfolio_cap)": SB.RISK_LEVELS, "ticker_cap": SB.TICKER_CAP,
                            "fee_per_contract_per_leg": SB.FEE_PER_CONTRACT, "half_spread_pct": SB.HALF_SPREAD_PCT,
                            "min_half_spread": {**SB.MIN_HALF, "default": SB.MIN_HALF_DEFAULT}, "stock_slip_bps": SB.STOCK_SLIP_BPS,
                            "default_jump": SB.DEFAULT_JUMP, "cycle": "enter first session of even weeks at open; expiry last session of next week"},
            "rejected_structures": __import__("Options.Structures", fromlist=["REJECTED"]).REJECTED,
            "nested": {str(a): {k: v for k, v in n.items() if k != "eq_central"} for a, n in nested.items()},
            "top_by_PRE_min_sharpe_10k": [t["variant"] for t in top10k], "elapsed_s": time.time() - t0,
            "units": {"equity": "USD", "drawdown_pct": "negative fraction", "greeks": "position $-greeks: delta shares-equiv, vega $/vol pt, theta $/cal day"}}
    (run_dir / f"{BASE}_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({"top10k": [(t["variant"], round(t["PRE_min_sharpe"], 3), round(t["central_ALL_cagr"], 3)) for t in top10k],
                      "nested": {str(a): {"central": n["central"], "worst": n["worst"]} for a, n in nested.items()}}, indent=1, default=str))
    print("DONE", run_dir, f"{time.time()-t0:.0f}s", flush=True)
