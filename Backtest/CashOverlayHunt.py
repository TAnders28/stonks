"""
Cash-overlay hunt on the three high-vol base strategies.
Pre-declared overlays chosen on IS; holdout untouched for evaluation.
"""
from __future__ import annotations
import csv, json, math, time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from Backtest.MultiAssetEngine import Portfolio, CostModel, summarize, CODE_VERSION
from Backtest.Strategies.HighVolFamilies import (
    cross_sectional_momentum, ts_momentum_volscale, sma, realized_vol,
)
from Backtest import export_charting as chart
from Backtest.HuntHighVol import load_bars, index_by_date, buy_hold_equal

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
UNIVERSE = ROOT / "data/universe/highvol_2026-11-08.json"

# Pre-declared overlay catalog (10 including none)
OVERLAY_NAMES = [
    "none",
    "vix_ge_25",
    "vix_jump_20_5d",
    "spy_rv20_pctl80",
    "spy_below_200",
    "spy_below_50",
    "breadth_above50_lt40",
    "strat_dd_12_reenter_6",
    "hyg_ief_below_ma50",
    "composite_2of4",
]


@dataclass
class OverlayState:
    name: str
    in_cash: bool = False
    strat_peak: float = 0.0


def precompute_features(calendar, histories, date_index, spy_bars, spy_idx, vix_bars, vix_idx,
                        hyg_bars, hyg_idx, ief_bars, ief_idx):
    """Walk calendar once; store per-date feature dicts (no lookahead)."""
    feats = {}
    # precompute SPY RV20 series on spy index
    spy_rv = [None] * len(spy_bars)
    for i in range(20, len(spy_bars)):
        spy_rv[i] = realized_vol([spy_bars[k]["close"] for k in range(i + 1)], 20)

    ief_by_date = {b["date"]: b["close"] for b in ief_bars}

    for date in calendar:
        spy_i = spy_idx.get(date)
        vix_i = vix_idx.get(date)
        hyg_i = hyg_idx.get(date)
        f = {
            "spy": None, "spy_sma50": None, "spy_sma200": None,
            "vix": None, "vix_5d_jump": None,
            "spy_rv20": None, "spy_rv20_pctl": None,
            "breadth_pct": None,
            "hyg_ief": None, "hyg_ief_ma50": None,
            "vote_vix25": False, "vote_spy200": False, "vote_breadth": False, "vote_rv80": False,
        }
        if spy_i is not None:
            closes = [spy_bars[k]["close"] for k in range(spy_i + 1)]
            f["spy"] = closes[-1]
            if spy_i >= 49:
                f["spy_sma50"] = sma(closes, 50)
            if spy_i >= 199:
                f["spy_sma200"] = sma(closes, 200)
            f["spy_rv20"] = spy_rv[spy_i]
            if spy_i >= 20 + 252 and spy_rv[spy_i] is not None:
                window = [spy_rv[j] for j in range(spy_i - 251, spy_i + 1) if spy_rv[j] is not None]
                if window:
                    cur = spy_rv[spy_i]
                    f["spy_rv20_pctl"] = sum(1 for x in window if x <= cur) / len(window)

        if vix_i is not None:
            f["vix"] = vix_bars[vix_i]["close"]
            if vix_i >= 5 and vix_bars[vix_i - 5]["close"] > 0:
                f["vix_5d_jump"] = vix_bars[vix_i]["close"] / vix_bars[vix_i - 5]["close"] - 1.0

        # breadth
        above = n = 0
        for sym, bars in histories.items():
            j = date_index[sym].get(date)
            if j is None or j < 50:
                continue
            closes = [bars[k]["close"] for k in range(j + 1)]
            s = sma(closes, 50)
            if s is None:
                continue
            n += 1
            if closes[-1] > s:
                above += 1
        if n >= 10:
            f["breadth_pct"] = above / n

        # HYG/IEF
        if hyg_i is not None and hyg_i >= 50:
            ratio_hist = []
            for b in hyg_bars[hyg_i - 49 : hyg_i + 1]:
                ic = ief_by_date.get(b["date"])
                if ic and ic > 0:
                    ratio_hist.append(b["close"] / ic)
            if len(ratio_hist) >= 50:
                f["hyg_ief"] = ratio_hist[-1]
                f["hyg_ief_ma50"] = sum(ratio_hist[-50:]) / 50

        f["vote_vix25"] = f["vix"] is not None and f["vix"] >= 25
        f["vote_spy200"] = f["spy_sma200"] is not None and f["spy"] < f["spy_sma200"]
        f["vote_breadth"] = f["breadth_pct"] is not None and f["breadth_pct"] < 0.40
        f["vote_rv80"] = f["spy_rv20_pctl"] is not None and f["spy_rv20_pctl"] >= 0.80
        feats[date] = f
    return feats


def go_cash_today(name: str, f: dict, equity_today: float, st: OverlayState) -> bool:
    if name == "none":
        return False
    if name == "vix_ge_25":
        return bool(f["vote_vix25"])
    if name == "vix_jump_20_5d":
        return f["vix_5d_jump"] is not None and f["vix_5d_jump"] >= 0.20
    if name == "spy_rv20_pctl80":
        return bool(f["vote_rv80"])
    if name == "spy_below_200":
        return bool(f["vote_spy200"])
    if name == "spy_below_50":
        return f["spy_sma50"] is not None and f["spy"] < f["spy_sma50"]
    if name == "breadth_above50_lt40":
        return bool(f["vote_breadth"])
    if name == "strat_dd_12_reenter_6":
        st.strat_peak = max(st.strat_peak, equity_today) if st.strat_peak > 0 else equity_today
        dd = (equity_today / st.strat_peak - 1.0) if st.strat_peak > 0 else 0.0
        if not st.in_cash:
            if dd <= -0.12:
                st.in_cash = True
        else:
            if dd >= -0.06:
                st.in_cash = False
        return st.in_cash
    if name == "hyg_ief_below_ma50":
        return f["hyg_ief"] is not None and f["hyg_ief_ma50"] is not None and f["hyg_ief"] < f["hyg_ief_ma50"]
    if name == "composite_2of4":
        votes = sum([f["vote_vix25"], f["vote_spy200"], f["vote_breadth"], f["vote_rv80"]])
        return votes >= 2
    raise ValueError(name)


def run_with_overlay(
    histories, date_index, calendar, signal_fn, cost, start_equity, max_weight,
    overlay_name, feats, start_date, end_date,
):
    dates = [d for d in calendar if start_date <= d <= end_date]
    port = Portfolio(start_equity, cost, max_weight=max_weight)
    pending = None
    st = OverlayState(name=overlay_name, strat_peak=start_equity)
    overlay_log = []
    switches = 0
    prev_cash = False

    for date in dates:
        opens, closes = {}, {}
        i_by_sym = {}
        for sym, bars in histories.items():
            j = date_index[sym].get(date)
            if j is None:
                continue
            i_by_sym[sym] = j
            opens[sym] = bars[j]["open"]
            closes[sym] = bars[j]["close"]

        if pending is not None:
            port.execute_targets(date, opens, pending)
            pending = None

        port.mark(date, closes if closes else {s: port.positions[s]["entry_px"] for s in port.positions})
        eq = port.curve[-1]["equity"]
        f = feats.get(date, {})

        go_cash = go_cash_today(overlay_name, f, eq, st) if f else False
        # strat_dd needs live state already handled inside go_cash_today

        if go_cash != prev_cash:
            switches += 1
            prev_cash = go_cash

        dd = None
        if st.strat_peak > 0:
            dd = eq / st.strat_peak - 1.0

        overlay_log.append({
            "date": date,
            "in_cash": 1 if go_cash else 0,
            "breadth_pct": f.get("breadth_pct", ""),
            "vix": f.get("vix", ""),
            "spy": f.get("spy", ""),
            "spy_rv20_pctl": f.get("spy_rv20_pctl", ""),
            "votes": sum([f.get("vote_vix25"), f.get("vote_spy200"), f.get("vote_breadth"), f.get("vote_rv80")]) if f else "",
            "strat_dd": dd if dd is not None else "",
            "hyg_ief": f.get("hyg_ief", ""),
        })

        pending = {} if go_cash else signal_fn(histories, i_by_sym, True)

    cash_days = sum(1 for r in overlay_log if r["in_cash"] == 1)
    return port.curve, port.trades, overlay_log, {
        "pct_time_cash": cash_days / len(overlay_log) if overlay_log else 0.0,
        "n_switches": switches,
    }


def metrics(curve, trades, start_equity, n_bars, extra):
    s = summarize(curve, trades, start_equity, n_bars)
    s.update(extra)
    return s


def main():
    t0 = time.time()
    u = json.loads(UNIVERSE.read_text())
    exclude = {"leveraged_inverse", "benchmarks", "vol_proxy"}
    trade_buckets, cost_bps = {}, {}
    for bname, b in u["buckets"].items():
        if bname in exclude:
            continue
        syms = []
        for s in b["symbols"]:
            bars = load_bars(s)
            if bars and len(bars) >= 100:
                syms.append(s)
                cost_bps[s] = b.get("default_cost_bps", 12)
        if syms:
            trade_buckets[bname] = syms
    all_syms = sorted({s for v in trade_buckets.values() for s in v})
    histories = {s: load_bars(s) for s in all_syms}
    date_index = {s: index_by_date(histories[s]) for s in all_syms}

    spy_bars = load_bars("SPY"); spy_idx = index_by_date(spy_bars)
    vix_bars = load_bars("VIX"); vix_idx = index_by_date(vix_bars)
    hyg_bars = load_bars("HYG"); hyg_idx = index_by_date(hyg_bars)
    ief_bars = load_bars("IEF"); ief_idx = index_by_date(ief_bars)
    bil_bars = load_bars("BIL")

    calendar = [b["date"] for b in spy_bars if "2018-01-01" <= b["date"] <= "2026-11-06"]
    cut_i = int(len(calendar) * 0.80)
    is_start, is_end = calendar[0], calendar[cut_i - 1]
    ho_start, ho_end = calendar[cut_i], calendar[-1]
    print(f"IS {is_start}->{is_end} n={cut_i} | HO {ho_start}->{ho_end} n={len(calendar)-cut_i}")

    print("Precomputing overlay features...", flush=True)
    feats = precompute_features(
        calendar, histories, date_index, spy_bars, spy_idx, vix_bars, vix_idx,
        hyg_bars, hyg_idx, ief_bars, ief_idx,
    )
    print(f"Features ready in {time.time()-t0:.1f}s", flush=True)

    cost = CostModel(default_bps=12.0, symbol_bps=cost_bps)
    start_equity = 10_000.0
    max_w = 0.50
    bases = [
        ("xs_mom_126_top3", cross_sectional_momentum(126, 3, max_w, skip_days=1)),
        ("xs_mom_63_top3", cross_sectional_momentum(63, 3, max_w, skip_days=1)),
        ("ts_mom_f20_s100_tv20", ts_momentum_volscale(20, 100, 20, 0.20, max_w)),
    ]

    run_at = datetime.now(ET)
    run_id = f"hvo_{run_at.strftime('%Y%m%dT%H%M%S')}"
    run_dir = OUT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    base_name = f"{run_at.strftime('%Y-%m-%d')}_highvol-cash-overlay"

    trials = []
    for bname, sig in bases:
        print(f"\n=== BASE {bname} ===", flush=True)
        for oname in OVERLAY_NAMES:
            print(f"  {oname} ...", flush=True)
            cis, tis, _, xis = run_with_overlay(
                histories, date_index, calendar, sig, cost, start_equity, max_w,
                oname, feats, is_start, is_end,
            )
            cho, tho, _, xho = run_with_overlay(
                histories, date_index, calendar, sig, cost, start_equity, max_w,
                oname, feats, ho_start, ho_end,
            )
            mis = metrics(cis, tis, start_equity, cut_i, xis)
            mho = metrics(cho, tho, start_equity, len(calendar) - cut_i, xho)
            trials.append({"base": bname, "overlay": oname, "insample": mis, "holdout": mho})
            print(
                f"    IS cagr={mis['cagr']*100:.1f}% sh={mis['sharpe']:.2f} dd={mis['max_drawdown']*100:.1f}% cash={mis['pct_time_cash']*100:.0f}% sw={mis['n_switches']} | "
                f"HO cagr={mho['cagr']*100:.1f}% sh={mho['sharpe']:.2f} dd={mho['max_drawdown']*100:.1f}% cash={mho['pct_time_cash']*100:.0f}%",
                flush=True,
            )

    n_trials = len(trials)
    picks = {}
    for bname, _ in bases:
        subset = [t for t in trials if t["base"] == bname]
        none = next(t for t in subset if t["overlay"] == "none")
        none_sh = none["insample"]["sharpe"]
        none_dd = none["insample"]["max_drawdown"]
        scored = []
        for t in subset:
            sh = t["insample"]["sharpe"]
            dd = t["insample"]["max_drawdown"]
            if sh != sh:
                continue
            if t["overlay"] != "none" and sh < none_sh - 0.05:
                continue
            reward = 0.5 * max(0.0, dd - none_dd)  # less-negative DD is better
            score = sh + reward
            scored.append((score, t))
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]
        picks[bname] = {
            "best": best,
            "none": none,
            "is_selection_score": scored[0][0],
            "ranked_is": [
                {"overlay": t["overlay"], "score": sc, "is_sharpe": t["insample"]["sharpe"],
                 "is_dd": t["insample"]["max_drawdown"], "is_cash": t["insample"]["pct_time_cash"]}
                for sc, t in scored[:6]
            ],
        }

    series_curves = {}
    all_trades = []
    overlay_state_rows = []
    current_states = {}

    for bname, sig in bases:
        wanted = []
        wanted.append(("none", "none"))
        best_o = picks[bname]["best"]["overlay"]
        if best_o != "none":
            wanted.append(("best", best_o))
        # also always include composite and spy_below_200 as reference overlays if not already
        for ref in ("composite_2of4", "spy_below_200", "strat_dd_12_reenter_6"):
            if ref != best_o and ref != "none":
                wanted.append((ref, ref))
        seen = set()
        for tag, oname in wanted:
            series = f"{bname}__{oname}"
            if series in seen:
                continue
            seen.add(series)
            curve, trades, olog, _ = run_with_overlay(
                histories, date_index, calendar, sig, cost, start_equity, max_w,
                oname, feats, calendar[0], calendar[-1],
            )
            series_curves[series] = [{"date": e["date"], "equity": e["equity"]} for e in curve]
            for tr in trades:
                all_trades.append({
                    "entry_date": tr.entry_date, "exit_date": tr.exit_date, "symbol": tr.symbol,
                    "side": tr.side, "qty": tr.qty, "entry_price": tr.entry_price,
                    "exit_price": tr.exit_price, "pnl": tr.pnl,
                })
            for row in olog:
                overlay_state_rows.append({"series": series, **row})
            current_states[series] = {
                "as_of": olog[-1]["date"],
                "in_cash": bool(olog[-1]["in_cash"]),
                "overlay": oname,
                "vix": olog[-1].get("vix"),
                "breadth_pct": olog[-1].get("breadth_pct"),
                "votes": olog[-1].get("votes"),
            }

    spy_curve, _ = buy_hold_equal(
        {"SPY": spy_bars}, {"SPY": spy_idx}, calendar, ["SPY"], CostModel(5), start_equity, 1.0,
        calendar[0], calendar[-1],
    )
    series_curves["SPY_buy_hold"] = [{"date": e["date"], "equity": e["equity"]} for e in spy_curve]

    chart.write_equity_csv(run_dir / f"{base_name}_equity.csv", series_curves)
    with (run_dir / f"{base_name}_trades.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["entry_date","exit_date","symbol","side","qty","entry_price","exit_price","pnl"])
        w.writeheader(); w.writerows(all_trades)

    state_fields = ["date","series","in_cash","breadth_pct","vix","spy","spy_rv20_pctl","votes","strat_dd","hyg_ief"]
    with (run_dir / f"{base_name}_overlay-state.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=state_fields, extrasaction="ignore")
        w.writeheader()
        for row in overlay_state_rows:
            w.writerow({k: row.get(k, "") for k in state_fields})

    meta = {
        "base_name": base_name,
        "run_id": run_id,
        "code_version": CODE_VERSION + "+cash-overlay",
        "run_at_et": run_at.isoformat(timespec="seconds"),
        "timezone": "America/New_York",
        "units": {
            "equity": "USD",
            "drawdown_pct": "negative fraction vs peak",
            "in_cash": "1=flat/cash, 0=risk-on",
            "pnl": "USD",
            "dates": "ISO 8601 date",
        },
        "simulated_vs_real": "simulated",
        "fees_included": True,
        "cash_yield": "0 (BIL not credited)",
        "bil_note": f"BIL {bil_bars[-1]['close']} on {bil_bars[-1]['date']}" if bil_bars else "n/a",
        "execution": "overlay_and_signal_on_close_act_next_open",
        "insample": f"{is_start}..{is_end}",
        "holdout": f"{ho_start}..{ho_end}",
        "n_trials": n_trials,
        "overlays_predeclared": OVERLAY_NAMES,
        "selection_rule_is": "Among overlays with IS Sharpe >= none_IS_Sharpe - 0.05, maximize IS_Sharpe + 0.5*max(0, IS_maxDD - none_IS_maxDD)",
        "bases": [b for b, _ in bases],
        "files": [
            f"{base_name}_equity.csv", f"{base_name}_trades.csv",
            f"{base_name}_overlay-state.csv", f"{base_name}_meta.json",
            "REPORT.md", "results.json",
        ],
    }
    chart.write_meta(run_dir / f"{base_name}_meta.json", meta)

    results = {
        "meta": meta,
        "trials": trials,
        "picks": {
            k: {
                "best_overlay": v["best"]["overlay"],
                "is_selection_score": v["is_selection_score"],
                "ranked_is": v["ranked_is"],
                "none_insample": v["none"]["insample"],
                "none_holdout": v["none"]["holdout"],
                "best_insample": v["best"]["insample"],
                "best_holdout": v["best"]["holdout"],
            } for k, v in picks.items()
        },
        "current_states": current_states,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))

    def pct(x):
        return "n/a" if x != x else f"{x*100:.2f}%"
    def shv(x):
        return "n/a" if x != x else f"{x:.2f}"

    lines = [
        f"# High-vol cash overlay — `{run_id}`",
        "",
        f"- Run at (ET): **{meta['run_at_et']}** · freeze / research only",
        f"- IS `{is_start}→{is_end}` · Holdout `{ho_start}→{ho_end}`",
        f"- **{n_trials} trials** = 3 bases × {len(OVERLAY_NAMES)} pre-declared overlays",
        f"- Cash earns **0** (BIL last {meta['bil_note']} not credited)",
        f"- Act at next open; costs on switches",
        f"- IS selection: {meta['selection_rule_is']}",
        "",
        "## Overlay catalog",
        "",
    ]
    desc = {
        "none": "always risk-on",
        "vix_ge_25": "cash if VIX ≥ 25",
        "vix_jump_20_5d": "cash if VIX up ≥20% over 5 sessions",
        "spy_rv20_pctl80": "cash if SPY 20d RV ≥ 80th pctl of trailing 252",
        "spy_below_200": "cash if SPY < SMA200",
        "spy_below_50": "cash if SPY < SMA50",
        "breadth_above50_lt40": "cash if <% of universe above SMA50 is <40%",
        "strat_dd_12_reenter_6": "cash after strategy DD ≤ −12% from peak; re-enter at −6%",
        "hyg_ief_below_ma50": "cash if HYG/IEF < its 50d MA (credit stress)",
        "composite_2of4": "cash if ≥2 of {VIX≥25, SPY<200, breadth<40%, RV pctl≥80}",
    }
    for n in OVERLAY_NAMES:
        lines.append(f"- `{n}`: {desc[n]}")
    lines.append("")

    for bname, _ in bases:
        pk = picks[bname]
        none, best = pk["none"], pk["best"]
        haircut = best["holdout"]["sharpe"] / math.sqrt(math.log(n_trials)) if best["holdout"]["sharpe"] == best["holdout"]["sharpe"] else float("nan")
        lines += [
            f"## `{bname}`",
            "",
            f"**IS-selected overlay:** `{best['overlay']}`",
            "",
            "| Version | Window | CAGR | Sharpe | Sortino | MaxDD | % cash | Switches |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for label, t in [("no overlay", none), (f"best (`{best['overlay']}`)", best)]:
            for window, key in [("IS", "insample"), ("HO", "holdout")]:
                m = t[key]
                lines.append(
                    f"| {label} | {window} | {pct(m['cagr'])} | {shv(m['sharpe'])} | {shv(m['sortino'])} | "
                    f"{pct(m['max_drawdown'])} | {pct(m['pct_time_cash'])} | {m['n_switches']} |"
                )
        lines += [
            "",
            f"- Haircut HO Sharpe (best / sqrt(log({n_trials}))): **{shv(haircut)}**",
            f"- IS shortlist: " + ", ".join(f"`{x['overlay']}`({x['score']:.2f})" for x in pk["ranked_is"]),
            "",
        ]

    lines += ["## Current state (latest bar)", ""]
    for series, st in sorted(current_states.items()):
        mode = "CASH" if st["in_cash"] else "RISK-ON"
        lines.append(f"- `{series}`: **{mode}** as of {st['as_of']} (overlay=`{st['overlay']}`, VIX={st.get('vix')}, breadth={st.get('breadth_pct')}, votes={st.get('votes')})")

    lines += [
        "",
        "## Caveats",
        f"- {n_trials} trials; overlays picked on IS; holdout is the test.",
        "- Strategy-DD overlay uses live overlaid equity with hysteresis.",
        "- Survivorship / leveraged-ETF decay caveats unchanged.",
        "",
        f"## Files",
        f"`{run_dir}/`",
        f"- `{base_name}_equity.csv`",
        f"- `{base_name}_trades.csv`",
        f"- `{base_name}_overlay-state.csv`",
        f"- `{base_name}_meta.json`",
        "",
    ]
    (run_dir / "REPORT.md").write_text("\n".join(lines))
    print("\nDONE", run_dir, "elapsed", round(time.time()-t0,1))
    for bname, _ in bases:
        print("PICK", bname, "->", picks[bname]["best"]["overlay"],
              "HO sh", picks[bname]["best"]["holdout"]["sharpe"],
              "vs none", picks[bname]["none"]["holdout"]["sharpe"])


if __name__ == "__main__":
    main()
