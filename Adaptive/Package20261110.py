"""Package ap_20261110T051807 results into AgentDocs with prereg success test + Holm over the 2 primary tests."""
import csv, json, shutil, hashlib
from pathlib import Path
SRC = Path(r"E:\Stocks\stonk-data\out\ap_20261110T051807"); OLD = "2026-11-09_adaptive-market-policy"
DST = Path(r"E:\AgentDocs\Optimus-Stonk\2026-11-10_adaptive-market-policy"); NEW = "2026-11-10_adaptive-market-policy"
DST.mkdir(parents=True, exist_ok=True)
rd = lambda n: list(csv.DictReader(open(SRC / f"{OLD}_{n}.csv", encoding="utf-8")))
comp, prim = rd("comparison"), rd("primary_tests")
for n in ("comparison", "primary_tests", "allocations", "transitions", "trades", "arm_labels"):
    shutil.copyfile(SRC / f"{OLD}_{n}.csv", DST / f"{NEW}_{n}.csv")
shutil.copyfile(SRC / f"{OLD}_equity.csv", DST / f"{NEW}.csv")   # tidy date,series,equity,drawdown_pct
shutil.copyfile(SRC / f"{OLD}_meta.json", DST / f"{NEW}_run-meta.json")
shutil.copyfile(Path(r"E:\Stocks\stonk-data\Adaptive") / f"{OLD}_hypotheses.json", DST / f"{NEW}_preregistration.json")
shutil.copyfile(Path(r"E:\Stocks\stonk-data\Adaptive\Policy.py"), DST / f"{NEW}_Policy.py.txt")
KEY = ["A_table", "B_ridge", "annual_selector", "fixed_equity", "fixed_bull_call_debit", "fixed_iron_condor", "fixed_mix", "spy_cash_budget", "spy_100_bench", "spy_samevol_EXPOST_B_ridge"]
M = {(r["scenario"], r["series"]): r for r in comp}
cagr = lambda s, n: float(M[(s, n)]["cagr"])
P = {(r["scenario"], r["policy"]): r for r in prim}
res = {}
for pol in ("A_table", "B_ridge"):
    r = P[("central", pol)]
    p_raw = max(float(r["boot_p_vs_fixed_equity"]), float(r["boot_p_vs_spy_cash_budget"]))  # must beat BOTH -> binding = max
    beats = lambda s: cagr(s, pol) > max(cagr(s, "fixed_equity"), cagr(s, "spy_cash_budget"))
    res[pol] = {"central_cagr_beats_both": beats("central"), "boot_p_binding": p_raw, "perm_p": float(r["perm_p"]),
                "direction_cost3x": beats("cost3x"), "direction_phase1": beats("phase1")}
order = sorted(res, key=lambda k: res[k]["boot_p_binding"]); run = 0.0
for i, k in enumerate(order):
    run = max(run, min(1.0, (len(order) - i) * res[k]["boot_p_binding"])); res[k]["holm_p"] = round(run, 4)
for k, v in res.items():
    v["passes_prereg_success"] = bool(v["central_cagr_beats_both"] and v["holm_p"] < 0.05 and v["perm_p"] < 0.05 and v["direction_cost3x"] and v["direction_phase1"])
occ = {pol: {a: int(P[("central", pol)][f"occ_{a}"]) for a in ("cash", "equity", "defensive", "bull_call_debit", "bear_put_debit", "iron_condor")} for pol in res}
summary = {"base": NEW, "source_run": str(SRC), "hypotheses_sha256": hashlib.sha256((DST / f"{NEW}_preregistration.json").read_bytes()).hexdigest(),
           "policy_sha256": hashlib.sha256((DST / f"{NEW}_Policy.py.txt").read_bytes()).hexdigest(),
           "eval_window": [M[("central", "A_table")]["eval_start"], M[("central", "A_table")]["eval_end"]], "start_equity": 10000, "deposits": 0,
           "primary_tests": res, "verdict": "no demonstrated edge" if not any(v["passes_prereg_success"] for v in res.values()) else "prereg success criterion met",
           "central_occupancy_periods": occ,
           "metrics": {s: {n: {k: M[(s, n)][k] for k in ("end", "cagr", "maxdd", "vol", "sharpe_rf2", "worst_2w", "switches", "infeasible", "opt_entries", "day_trades")}
                           for n in KEY} for s in ("central", "cost3x", "phase1", "cost1x", "cost0_tickonly", "iv_rv1.00", "iv_rv1.10", "iv_rv1.25", "itm0.25", "budget0.05")}}
(DST / f"{NEW}.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({"primary_tests": res, "verdict": summary["verdict"]}, indent=1))
