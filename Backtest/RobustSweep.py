"""Momentum robustness sweep + rolling-origin walk-forward + earnings-skip test.

Pre-declared (written before running):
  Grid: lb {63,126,189,252} x top {3,5,8} x stop {0,10%,15%,20%} x reb {W,2W,M} x overlay {none,spy200,vt40,spy200_vt40}
  Windows: EARLY 2010-01-04..2017-12-29 (validation), IS 2018-01-02..2025-01-30, HO 2025-01-31..END (contaminated: peeked before).
  Selection uses EARLY+IS only:
     score = min(Sharpe_EARLY, Sharpe_IS) + 0.75 * min(MaxDD_EARLY, MaxDD_IS)      (MaxDD negative)
     robust_score = mean(score over the config and its one-step grid neighbours, same overlay)
     pick argmax robust_score.
  Walk-forward: for test year Y=2013..2026, select by the same plateau rule using only 2010-01-04..(Y-1)-12-31
     (score_train = Sharpe + 0.75*MaxDD on the train span), trade year Y. Stitched OOS curve reported.
  Earnings skip: tested on the selected rule and the prior rule only (2 extra trials). Coverage ~2018+.
Every trial is appended to out/trial_log.jsonl. Haircut Sharpe = Sharpe / sqrt(ln N_trials).

Usage: .venv\\Scripts\\python -m Backtest.RobustSweep --end 2026-11-06 --procs 20
"""
from __future__ import annotations
import argparse, csv, itertools, json, math, time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np

from Backtest.VecMomentum import (load_panel, build_features, simulate, window_metrics, load_earnings_block, CODE_VERSION)

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
LBS = [63, 126, 189, 252]; TOPS = [3, 5, 8]; STOPS = [0.0, 0.10, 0.15, 0.20]; REBS = ["W", "2W", "M"]
OVS = ["none", "spy200", "vt40", "spy200_vt40"]
PRIOR = {"lb": 126, "top": 5, "stop": 0.15, "reb": "W", "ov": "none"}
DD_W = 0.75
_G = {}


def cfg_name(c):
    s = f"xs{c['lb']}_top{c['top']}_stop{int(round(c['stop']*100))}_{c['reb']}_{c['ov']}"
    return s + ("_earnskip" if c.get("earn") else "")


def _init(end):
    P = load_panel(start="2007-01-01", end=end)
    _G["P"] = P; _G["F"] = build_features(P)
    d = P.dates
    _G["s0"] = next(i for i, x in enumerate(d) if x >= "2010-01-01")
    _G["e0"] = len(d) - 1


def _run(cfg):
    P, F = _G["P"], _G["F"]
    r = simulate(P, F, cfg, _G["s0"], _G["e0"])
    return cfg, r["eq"].astype(np.float64), r["expo"].astype(np.float32), r["entries"], r["turn"].astype(np.float32)


def idx_range(dates, a, b):
    s = next(i for i, x in enumerate(dates) if x >= a)
    e = max(i for i, x in enumerate(dates) if x <= b)
    return s, e


def neighbours(c):
    out = []
    for key, vals in (("lb", LBS), ("top", TOPS), ("stop", STOPS), ("reb", REBS)):
        i = vals.index(c[key])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                n = dict(c); n[key] = vals[j]; out.append(n)
    return out


def key(c):
    return (c["lb"], c["top"], c["stop"], c["reb"], c["ov"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default="2026-11-06")
    ap.add_argument("--procs", type=int, default=20)
    ap.add_argument("--slug", default="momentum-robustness")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    run_at = datetime.now(ET)
    run_id = f"rs_{run_at.strftime('%Y%m%dT%H%M%S')}"
    run_dir = OUT / run_id; run_dir.mkdir(parents=True, exist_ok=False)
    base = f"{run_at.strftime('%Y-%m-%d')}_{args.slug}"
    grid = [dict(lb=a, top=b, stop=c, reb=d, ov=e) for a, b, c, d, e in itertools.product(LBS, TOPS, STOPS, REBS, OVS)]
    if args.quick:
        grid = [g for g in grid if g["ov"] in ("none",) and g["reb"] == "W" and g["stop"] in (0.0, 0.15)]
    _init(args.end)
    P, F = _G["P"], _G["F"]; dates = P.dates
    W = {"EARLY": idx_range(dates, "2010-01-01", "2017-12-31"), "IS": idx_range(dates, "2018-01-01", "2025-01-30"),
         "HO": idx_range(dates, "2025-01-31", args.end), "IS19": idx_range(dates, "2019-01-01", "2025-01-30")}
    print(f"panel syms={len(P.syms)} dates {dates[0]}..{dates[-1]} sim from {dates[_G['s0']]}; trials={len(grid)}", flush=True)
    with Pool(args.procs, initializer=_init, initargs=(args.end,)) as pool:
        res = pool.map(_run, grid, chunksize=4)
    print(f"grid done {time.time()-t0:.0f}s", flush=True)
    rows, curves = [], {}
    for cfg, eq, expo, ent, turn in res:
        m = {w: window_metrics(eq, expo, ent, *W[w]) for w in W}
        sc = min(m["EARLY"]["sharpe"], m["IS"]["sharpe"]) + DD_W * min(m["EARLY"]["maxdd"], m["IS"]["maxdd"])
        rows.append({"cfg": cfg, "m": m, "score": sc, "turn_ann": float(turn[_G['s0']:].sum() / ((len(turn) - _G['s0']) / 252))})
        curves[key(cfg)] = eq
    by = {key(r["cfg"]): r for r in rows}
    for r in rows:
        nb = [by[key(n)]["score"] for n in neighbours(r["cfg"]) if key(n) in by]
        r["robust"] = float(np.mean([r["score"]] + nb))
    rows.sort(key=lambda r: r["robust"], reverse=True)
    sel = rows[0]; sel_cfg = sel["cfg"]
    prior_r = by.get(key(PRIOR))
    n_grid = len(rows)

    # ---------- walk-forward (rolling origin, expanding) ----------
    wf = []; wf_ret = np.full(len(dates), np.nan)
    s0 = _G["s0"]
    for Y in range(2013, int(args.end[:4]) + 1):
        tr = idx_range(dates, "2010-01-01", f"{Y-1}-12-31")
        te = idx_range(dates, f"{Y}-01-01", min(f"{Y}-12-31", args.end))
        sc = {}
        for r in rows:
            mm = window_metrics(curves[key(r["cfg"])], np.zeros(len(dates)), np.zeros(len(dates), dtype=int), *tr)
            sc[key(r["cfg"])] = mm["sharpe"] + DD_W * mm["maxdd"]
        rob = {k: np.mean([sc[k]] + [sc[key(n)] for n in neighbours(by[k]["cfg"]) if key(n) in sc]) for k in sc}
        kbest = max(rob, key=rob.get)
        eq = curves[kbest]
        wf_ret[te[0]:te[1] + 1] = eq[te[0]:te[1] + 1] / eq[te[0] - 1:te[1]] - 1
        mm = window_metrics(eq, np.zeros(len(dates)), np.zeros(len(dates), dtype=int), *te)
        wf.append({"test_year": Y, "chosen": cfg_name(by[kbest]["cfg"]), "train_robust_score": rob[kbest], **{k: mm[k] for k in ("cagr", "sharpe", "maxdd")}})
        print(f"WF {Y}: {cfg_name(by[kbest]['cfg'])} -> CAGR {mm['cagr']*100:.1f}% Sh {mm['sharpe']:.2f} DD {mm['maxdd']*100:.1f}%", flush=True)
    wf_start = idx_range(dates, "2013-01-01", args.end)[0]
    wf_eq = np.full(len(dates), np.nan); wf_eq[wf_start - 1] = 10_000.0
    for t in range(wf_start, len(dates)):
        wf_eq[t] = wf_eq[t - 1] * (1 + wf_ret[t])

    # ---------- earnings-skip trials ----------
    earn = {}
    for nm, c in (("selected", sel_cfg), ("prior", PRIOR)):
        blk, cov = load_earnings_block(P, F["reb"][c["reb"]])
        r_e = simulate(P, F, c, s0, len(dates) - 1, earn_block=blk)
        ce = dict(c); ce["earn"] = True
        earn[nm] = {"cfg": ce, "eq": r_e["eq"], "m": {w: window_metrics(r_e["eq"], r_e["expo"], r_e["entries"], *W[w]) for w in W},
                    "coverage_frac_2019on": float(cov[W["IS19"][0]:].mean()), "blocked_frac_2019on": float(blk[W["IS19"][0]:][F["reb"][c["reb"]][W["IS19"][0]:]].mean())}
    n_trials = n_grid + len(earn)
    hc = math.sqrt(math.log(n_trials))

    # ---------- detailed run of selected (trades) ----------
    det = simulate(P, F, sel_cfg, s0, len(dates) - 1, record=True)
    det_prior = simulate(P, F, PRIOR, s0, len(dates) - 1, record=True)
    spy_eq = P.spy_c / P.spy_c[s0 - 1] * 10_000.0
    zeros = np.zeros(len(dates)); zi = np.zeros(len(dates), dtype=int)
    spy_m = {w: window_metrics(spy_eq, np.ones(len(dates)), zi, *W[w]) for w in W}
    wf_m = {w: window_metrics(wf_eq, zeros, zi, max(W[w][0], wf_start), W[w][1]) for w in ("IS", "HO")}
    wf_m["2013-2017"] = window_metrics(wf_eq, zeros, zi, wf_start, W["EARLY"][1])
    wf_m["ALL_2013on"] = window_metrics(wf_eq, zeros, zi, wf_start, len(dates) - 1)

    # ---------- write outputs ----------
    with (OUT / "trial_log.jsonl").open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"run_id": run_id, "logged_at_et": run_at.isoformat(timespec="seconds"), "name": cfg_name(r["cfg"]),
                                "cfg": r["cfg"], "score": r["score"], "robust": r["robust"],
                                **{f"{w}_{k}": r["m"][w][k] for w in W for k in ("cagr", "sharpe", "maxdd")}}) + "\n")
        for nm, e in earn.items():
            f.write(json.dumps({"run_id": run_id, "logged_at_et": run_at.isoformat(timespec="seconds"), "name": cfg_name(e["cfg"]), "cfg": e["cfg"],
                                **{f"{w}_{k}": e["m"][w][k] for w in W for k in ("cagr", "sharpe", "maxdd")}}) + "\n")
    with (run_dir / f"{base}_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "name", "lb", "top", "stop", "reb", "overlay", "score", "robust_score", "turnover_ann"] +
                   [f"{x}_{k}" for x in ("EARLY", "IS", "HO") for k in ("cagr", "sharpe", "maxdd", "expo")])
        for i, r in enumerate(rows):
            c = r["cfg"]
            w.writerow([i + 1, cfg_name(c), c["lb"], c["top"], c["stop"], c["reb"], c["ov"], round(r["score"], 4), round(r["robust"], 4), round(r["turn_ann"], 2)] +
                       [round(r["m"][x][k], 4) for x in ("EARLY", "IS", "HO") for k in ("cagr", "sharpe", "maxdd", "expo")])

    def dd(eq):
        pk = np.fmax.accumulate(np.where(np.isfinite(eq), eq, -np.inf)); return eq / pk - 1
    series = {"recommended": det["eq"], "prior_rule": det_prior["eq"], "recommended_earnskip": earn["selected"]["eq"],
              "walkforward_oos": wf_eq, "spy_buyhold": spy_eq}
    with (run_dir / f"{base}_equity.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["date", "series", "equity", "drawdown_pct"])
        for nm, eq in series.items():
            d_ = dd(eq)
            for t in range(s0 - 1 if nm != "walkforward_oos" else wf_start - 1, len(dates)):
                if np.isfinite(eq[t]):
                    w.writerow([dates[t], nm, round(float(eq[t]), 2), round(float(d_[t]), 5)])
    with (run_dir / f"{base}_trades.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["entry_date", "exit_date", "symbol", "side", "qty", "entry_price", "exit_price", "pnl"])
        for tr in det["trades"]:
            w.writerow([tr["entry_date"], tr["exit_date"], tr["symbol"], tr["side"], round(tr["qty"], 6), round(tr["entry_price"], 4), round(tr["exit_price"], 4), round(tr["pnl"], 2)])
    with (run_dir / f"{base}_walkforward.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["test_year", "chosen", "train_robust_score", "cagr", "sharpe", "maxdd"])
        for r in wf:
            w.writerow([r["test_year"], r["chosen"], round(r["train_robust_score"], 4), round(r["cagr"], 4), round(r["sharpe"], 3), round(r["maxdd"], 4)])
    met_rows = []
    def add(series, mm, haircut=True):
        for win, x in mm.items():
            met_rows.append({"series": series, "window": win, "cagr": x["cagr"], "sharpe": x["sharpe"],
                             "haircut_sharpe": x["sharpe"] / hc if haircut else x["sharpe"], "maxdd": x["maxdd"], "expo": x.get("expo")})
    add(cfg_name(sel_cfg) + " (recommended)", sel["m"])
    add(cfg_name(PRIOR) + " (prior, new engine)", prior_r["m"])
    for nm, e in earn.items():
        add(cfg_name(e["cfg"]), e["m"])
    add("walkforward_oos", wf_m)
    add("SPY buy&hold", spy_m, haircut=False)
    with (run_dir / f"{base}_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["series", "window", "cagr", "sharpe", "haircut_sharpe", "maxdd", "expo"]); w.writeheader()
        for r in met_rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    # plateau neighbourhood of selected
    nb_rows = [by[key(n)] for n in neighbours(sel_cfg) if key(n) in by]
    summary = {
        "run_id": run_id, "base_name": base, "run_at_et": run_at.isoformat(timespec="seconds"), "code_version": CODE_VERSION,
        "end": args.end, "panel_symbols": P.syms, "n_symbols": len(P.syms), "windows": {k: [dates[v[0]], dates[v[1]]] for k, v in W.items()},
        "n_grid_trials": n_grid, "n_trials_total": n_trials, "haircut_divisor": hc,
        "selection_rule": f"score=min(Sh_EARLY,Sh_IS)+{DD_W}*min(DD_EARLY,DD_IS); robust=mean over self+1-step neighbours; argmax",
        "selected": {"name": cfg_name(sel_cfg), "cfg": sel_cfg, "score": sel["score"], "robust": sel["robust"], "metrics": sel["m"]},
        "prior": {"name": cfg_name(PRIOR), "cfg": PRIOR, "score": prior_r["score"], "robust": prior_r["robust"], "metrics": prior_r["m"],
                  "rank": rows.index(prior_r) + 1},
        "neighbours_of_selected": [{"name": cfg_name(n["cfg"]), "score": n["score"], **{f"{w}_sharpe": n["m"][w]["sharpe"] for w in W}, **{f"{w}_maxdd": n["m"][w]["maxdd"] for w in W}} for n in nb_rows],
        "top10": [{"name": cfg_name(r["cfg"]), "robust": r["robust"], "score": r["score"], **{f"{w}_{k}": r["m"][w][k] for w in ("EARLY", "IS", "HO") for k in ("cagr", "sharpe", "maxdd")}} for r in rows[:10]],
        "earnings": {nm: {"name": cfg_name(e["cfg"]), "metrics": e["m"], "coverage_frac_2019on": e["coverage_frac_2019on"], "blocked_frac_2019on": e["blocked_frac_2019on"]} for nm, e in earn.items()},
        "walkforward": wf, "walkforward_metrics": wf_m, "spy": spy_m,
        "selected_final_targets": {P.syms[i]: float(v) for i, v in enumerate(det["last_targets"]) if v > 0},
        "elapsed_s": time.time() - t0,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({k: summary[k] for k in ("run_id", "n_trials_total", "selected", "prior")}, indent=1, default=float))
    print("DONE", run_dir, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
