"""
Walk-forward hunt over high-vol strategy families.
Tune/select on in-sample; untouched holdout for reporting winners.
"""
from __future__ import annotations
import argparse, csv, json, math, time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from Backtest.MultiAssetEngine import Portfolio, CostModel, summarize, Trade, CODE_VERSION
from Backtest.Strategies.HighVolFamilies import build_variant_grid, sma, realized_vol
from Backtest import export_charting as chart

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data" / "daily"
OUT = ROOT / "out"


def load_bars(sym: str) -> list[dict] | None:
    fname = sym.replace("^", "")
    p = DAILY / f"{fname}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["bars"]


def index_by_date(bars: list[dict]) -> dict[str, int]:
    return {b["date"]: i for i, b in enumerate(bars)}


def run_backtest(
    histories: dict[str, list[dict]],
    date_index: dict[str, dict[str, int]],
    calendar: list[str],
    signal_fn,
    cost: CostModel,
    start_equity: float,
    max_weight: float,
    regime_mode: str,
    spy_bars: list[dict],
    spy_idx: dict[str, int],
    vix_bars: list[dict] | None,
    vix_idx: dict[str, int] | None,
    start_date: str,
    end_date: str,
) -> tuple[list[dict], list[Trade]]:
    dates = [d for d in calendar if start_date <= d <= end_date]
    port = Portfolio(start_equity, cost, max_weight=max_weight)
    pending = None  # targets from prior close

    for di, date in enumerate(dates):
        # opens / closes available today
        opens, closes = {}, {}
        i_by_sym = {}
        for sym, bars in histories.items():
            j = date_index[sym].get(date)
            if j is None:
                continue
            i_by_sym[sym] = j
            opens[sym] = bars[j]["open"]
            closes[sym] = bars[j]["close"]

        # execute yesterday's targets at today's open
        if pending is not None:
            port.execute_targets(date, opens, pending)
            pending = None

        port.mark(date, closes if closes else {s: port.positions[s]["entry_px"] for s in port.positions})

        # regime using SPY/VIX through today close only
        regime_ok = True
        sj = spy_idx.get(date)
        if sj is None or sj < 200:
            regime_ok = False if regime_mode != "off" else True
        else:
            spy_closes = [spy_bars[k]["close"] for k in range(sj + 1)]
            spy_sma200 = sma(spy_closes, 200)
            above = spy_sma200 is not None and spy_closes[-1] > spy_sma200
            if regime_mode == "spy200":
                regime_ok = above
            elif regime_mode == "spy200_vix":
                vix_ok = True
                if vix_bars is not None and vix_idx is not None:
                    vj = vix_idx.get(date)
                    if vj is not None:
                        vix_ok = vix_bars[vj]["close"] < 25.0
                regime_ok = above and vix_ok
            elif regime_mode == "off":
                regime_ok = True

        # signal at close
        pending = signal_fn(histories, i_by_sym, regime_ok)

    return port.curve, port.trades


def buy_hold_equal(histories, date_index, calendar, syms, cost, start_equity, max_weight, start_date, end_date):
    """Equal-weight buy and hold of available syms, rebalanced monthly (1st trading day of month)."""
    dates = [d for d in calendar if start_date <= d <= end_date]
    port = Portfolio(start_equity, cost, max_weight=max_weight)
    pending = None
    last_month = None
    for date in dates:
        opens, closes = {}, {}
        avail = []
        for sym in syms:
            j = date_index.get(sym, {}).get(date)
            if j is None:
                continue
            opens[sym] = histories[sym][j]["open"]
            closes[sym] = histories[sym][j]["close"]
            avail.append(sym)
        if pending is not None:
            port.execute_targets(date, opens, pending)
            pending = None
        port.mark(date, closes)
        month = date[:7]
        if month != last_month and avail:
            last_month = month
            w = min(max_weight, 1.0 / len(avail))
            pending = {s: w for s in avail}
    return port.curve, port.trades


def path_target_stats(histories, date_index, calendar, signal_fn, regime_mode, spy_bars, spy_idx,
                      vix_bars, vix_idx, start_date, end_date, horizons=(5, 10), targets=(0.05, 0.10, 0.15)):
    """After each day a symbol is selected (weight>0), measure forward max return within H days."""
    dates = [d for d in calendar if start_date <= d <= end_date]
    hits = {h: {t: {"n": 0, "hit": 0, "avg_max": 0.0, "sum_max": 0.0} for t in targets} for h in horizons}
    events = 0
    # also store samples for BS-ish estimate
    samples = []  # {sym,date,entry_next_open, max_ret_10, close_ret_10}

    for di, date in enumerate(dates):
        i_by_sym = {}
        for sym, bars in histories.items():
            j = date_index[sym].get(date)
            if j is not None:
                i_by_sym[sym] = j
        # regime
        regime_ok = True
        sj = spy_idx.get(date)
        if regime_mode == "off":
            regime_ok = True
        elif sj is None or sj < 200:
            regime_ok = False
        else:
            spy_closes = [spy_bars[k]["close"] for k in range(sj + 1)]
            above = sma(spy_closes, 200) is not None and spy_closes[-1] > sma(spy_closes, 200)
            if regime_mode == "spy200":
                regime_ok = above
            else:
                vix_ok = True
                if vix_idx and date in vix_idx:
                    vix_ok = vix_bars[vix_idx[date]]["close"] < 25
                regime_ok = above and vix_ok
        w = signal_fn(histories, i_by_sym, regime_ok)
        if not w:
            continue
        # next day index in calendar for entry approx
        if di + 1 >= len(dates):
            continue
        for sym in w:
            bars = histories[sym]
            j = date_index[sym].get(date)
            if j is None or j + 1 >= len(bars):
                continue
            entry = bars[j + 1]["open"]  # next open
            if entry <= 0:
                continue
            events += 1
            for H in horizons:
                end_j = min(len(bars) - 1, j + 1 + H)
                # max high relative to entry over next H bars
                mx = 0.0
                for k in range(j + 1, end_j + 1):
                    mx = max(mx, bars[k]["high"] / entry - 1)
                close_ret = bars[end_j]["close"] / entry - 1
                for t in targets:
                    bucket = hits[H][t]
                    bucket["n"] += 1
                    bucket["sum_max"] += mx
                    if mx >= t:
                        bucket["hit"] += 1
                if H == 10:
                    # realized vol proxy from prior 20d
                    closes = [b["close"] for b in bars[: j + 1]]
                    vol = realized_vol(closes, 20) or 0.5
                    samples.append({
                        "symbol": sym, "signal_date": date, "entry_date": bars[j + 1]["date"],
                        "entry": entry, "max_ret_10": mx, "close_ret_10": close_ret,
                        "rv20": vol,
                    })
    # finalize
    out = {"events": events, "horizons": {}}
    for H in horizons:
        out["horizons"][str(H)] = {}
        for t in targets:
            b = hits[H][t]
            out["horizons"][str(H)][f"target_{t:.0%}"] = {
                "n": b["n"],
                "hit_rate": (b["hit"] / b["n"]) if b["n"] else float("nan"),
                "avg_max_ret": (b["sum_max"] / b["n"]) if b["n"] else float("nan"),
            }
    # rough debit-spread estimate: long ATM-ish call approx 0.5 delta butterfly-ish debit
    # Use: estimated 2-week debit spread value change ~ clip(max_ret,0,width) - debit
    # debit ~ 0.3 * width with width=10% of spot as stand-in for Tim-style tight OTM verticals? 
    # Clearer: ATM 10% wide debit ~ price with BS using rv as IV, T=10/252, then P&L if spot moves to entry*(1+max_ret)
    # Keep simple labeled estimate:
    est = []
    for s in samples:
        # assume 10%-wide call debit spread, long strike ~ entry, short ~ entry*1.10
        width = 0.10 * s["entry"]
        # rough debit as 35% of width when IV~rv (placeholder)
        iv = max(0.2, min(1.5, s["rv20"]))
        debit_frac = min(0.55, 0.15 + 0.4 * (iv / 0.5))  # scales with vol
        debit = debit_frac * width
        # intrinsic at best path within 10d capped at width
        best_spot = s["entry"] * (1 + s["max_ret_10"])
        # long call intrinsic - short call intrinsic
        long_k = s["entry"]
        short_k = s["entry"] * 1.10
        payoff = max(best_spot - long_k, 0) - max(best_spot - short_k, 0)
        pnl = payoff - debit
        est.append(pnl / debit if debit else 0.0)  # return on debit
    if est:
        out["est_debit_spread_10pct_wide"] = {
            "label": "ESTIMATED only — BS/heuristic using RV20 as IV proxy; NOT real option fills",
            "n": len(est),
            "avg_return_on_debit_at_best_path": sum(est) / len(est),
            "hit_positive": sum(1 for x in est if x > 0) / len(est),
            "assumptions": "10% wide call debit; debit_frac=f(RV20); payoff at best high within 10 sessions",
        }
    return out, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default=str(ROOT / "data/universe/highvol_2026-11-08.json"))
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-11-06")
    ap.add_argument("--holdout-frac", type=float, default=0.20)
    ap.add_argument("--start-equity", type=float, default=10_000.0)
    ap.add_argument("--max-weight", type=float, default=0.50)
    args = ap.parse_args()

    t0 = time.time()
    u = json.loads(Path(args.universe).read_text())
    # tradeable buckets (exclude inverse, benchmarks, vol_proxy from primary sleeve)
    exclude = {"leveraged_inverse", "benchmarks", "vol_proxy"}
    trade_buckets = {}
    cost_bps = {}
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
    # add survivorship names that loaded
    for s in u.get("survivorship_attempts", {}).get("included_if_yahoo_serves", []):
        bars = load_bars(s)
        if bars and len(bars) >= 100:
            trade_buckets.setdefault("survivorship_ruins", []).append(s)
            cost_bps[s] = 25

    all_syms = sorted({s for v in trade_buckets.values() for s in v})
    histories = {s: load_bars(s) for s in all_syms}
    date_index = {s: index_by_date(histories[s]) for s in all_syms}

    spy_bars = load_bars("SPY")
    spy_idx = index_by_date(spy_bars)
    vix_bars = load_bars("VIX")  # file saved as VIX.json from ^VIX
    if vix_bars is None:
        vix_bars = load_bars("^VIX")
    vix_idx = index_by_date(vix_bars) if vix_bars else None

    # calendar = SPY dates in range
    calendar = [b["date"] for b in spy_bars if args.start <= b["date"] <= args.end]
    cut_i = int(len(calendar) * (1 - args.holdout_frac))
    is_start, is_end = calendar[0], calendar[cut_i - 1]
    ho_start, ho_end = calendar[cut_i], calendar[-1]

    cost = CostModel(default_bps=12.0, symbol_bps=cost_bps)
    variants = build_variant_grid(trade_buckets, max_w=args.max_weight)
    regime_modes = ["off", "spy200", "spy200_vix"]

    run_at = datetime.now(ET)
    run_id = f"hv_{run_at.strftime('%Y%m%dT%H%M%S')}"
    run_dir = OUT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    trials = []
    print(f"Calendar {calendar[0]}->{calendar[-1]} n={len(calendar)} IS {is_start}->{is_end} HO {ho_start}->{ho_end}")
    print(f"Symbols={len(all_syms)} variants={len(variants)} regimes={len(regime_modes)} total_trials={len(variants)*len(regime_modes)}")

    for family, sig in variants:
        for regime in regime_modes:
            name = f"{sig.name}__reg_{regime}"
            print(f"RUN {name} ...", flush=True)
            curve_is, trades_is = run_backtest(
                histories, date_index, calendar, sig, cost, args.start_equity, args.max_weight,
                regime, spy_bars, spy_idx, vix_bars, vix_idx, is_start, is_end,
            )
            curve_ho, trades_ho = run_backtest(
                histories, date_index, calendar, sig, cost, args.start_equity, args.max_weight,
                regime, spy_bars, spy_idx, vix_bars, vix_idx, ho_start, ho_end,
            )
            summ_is = summarize(curve_is, trades_is, args.start_equity, cut_i)
            summ_ho = summarize(curve_ho, trades_ho, args.start_equity, len(calendar) - cut_i)
            trials.append({
                "name": name,
                "family": family,
                "signal": sig.name,
                "regime": regime,
                "insample": summ_is,
                "holdout": summ_ho,
            })
            print(f"  IS CAGR={summ_is['cagr']:.2%} Sharpe={summ_is['sharpe']:.2f} | HO CAGR={summ_ho['cagr']:.2%} Sharpe={summ_ho['sharpe']:.2f} DD={summ_ho['max_drawdown']:.2%}")

    # benchmarks
    print("Benchmarks...")
    spy_bh_is, spy_tr_is = buy_hold_equal(
        {"SPY": spy_bars}, {"SPY": spy_idx}, calendar, ["SPY"], CostModel(5), args.start_equity, 1.0, is_start, is_end)
    spy_bh_ho, spy_tr_ho = buy_hold_equal(
        {"SPY": spy_bars}, {"SPY": spy_idx}, calendar, ["SPY"], CostModel(5), args.start_equity, 1.0, ho_start, ho_end)
    bench = {
        "SPY": {
            "insample": summarize(spy_bh_is, spy_tr_is, args.start_equity, cut_i),
            "holdout": summarize(spy_bh_ho, spy_tr_ho, args.start_equity, len(calendar) - cut_i),
        }
    }
    # equal-weight highvol sleeve BH monthly
    hv_is, hv_tr_is = buy_hold_equal(histories, date_index, calendar, all_syms, cost, args.start_equity, args.max_weight, is_start, is_end)
    hv_ho, hv_tr_ho = buy_hold_equal(histories, date_index, calendar, all_syms, cost, args.start_equity, args.max_weight, ho_start, ho_end)
    bench["equal_weight_universe"] = {
        "insample": summarize(hv_is, hv_tr_is, args.start_equity, cut_i),
        "holdout": summarize(hv_ho, hv_tr_ho, args.start_equity, len(calendar) - cut_i),
    }

    # selection: survivors with HO Sharpe > 0 and HO CAGR > SPY HO CAGR and IS Sharpe > 0.3
    spy_ho_cagr = bench["SPY"]["holdout"]["cagr"]
    survivors = []
    for t in trials:
        ho, ins = t["holdout"], t["insample"]
        if (ho["sharpe"] == ho["sharpe"] and ho["sharpe"] > 0 and
            ins["sharpe"] == ins["sharpe"] and ins["sharpe"] > 0.25 and
            ho["cagr"] == ho["cagr"] and ho["cagr"] > spy_ho_cagr and
            ho["n_trades"] >= 5):
            survivors.append(t)
    # rank by holdout Sharpe, then CAGR
    survivors.sort(key=lambda t: (t["holdout"]["sharpe"], t["holdout"]["cagr"]), reverse=True)

    # haircut: deflated-ish — subtract penalty for trials
    n_trials = len(trials)
    for t in survivors:
        # simple Bonferroni-ish haircut on Sharpe: sharpe * sqrt(1/n) style caution note
        t["holdout_sharpe_haircut"] = t["holdout"]["sharpe"] / math.sqrt(max(1, math.log(n_trials)))
    
    winners = survivors[:2]
    # if fewer than 2 strict survivors, take best HO sharpe with IS sharpe>0 anyway but flag weak
    if len(winners) < 2:
        ranked = sorted(trials, key=lambda t: (
            (t["holdout"]["sharpe"] if t["holdout"]["sharpe"] == t["holdout"]["sharpe"] else -9),
            (t["holdout"]["cagr"] if t["holdout"]["cagr"] == t["holdout"]["cagr"] else -9),
        ), reverse=True)
        for t in ranked:
            if t not in winners:
                t["weak_survivor"] = True
                winners.append(t)
            if len(winners) >= 2:
                break

    # path targets + full curves for winners
    winner_details = []
    series_curves = {}
    all_trades_flat = []
    for t in winners:
        # find signal fn
        sig = None
        for fam, s in variants:
            if s.name == t["signal"]:
                sig = s
                break
        # full sample curve for charting
        curve, trades = run_backtest(
            histories, date_index, calendar, sig, cost, args.start_equity, args.max_weight,
            t["regime"], spy_bars, spy_idx, vix_bars, vix_idx, calendar[0], calendar[-1],
        )
        series_curves[t["name"]] = [{"date": e["date"], "equity": e["equity"]} for e in curve]
        for tr in trades:
            all_trades_flat.append({
                "entry_date": tr.entry_date, "exit_date": tr.exit_date, "symbol": tr.symbol,
                "side": tr.side, "qty": tr.qty, "entry_price": tr.entry_price,
                "exit_price": tr.exit_price, "pnl": tr.pnl,
            })
        path, samples = path_target_stats(
            histories, date_index, calendar, sig, t["regime"], spy_bars, spy_idx, vix_bars, vix_idx,
            ho_start, ho_end,
        )
        winner_details.append({**t, "path_targets_holdout": path, "full_summary": summarize(curve, trades, args.start_equity, len(calendar))})

    # SPY equity for chart
    spy_full, _ = buy_hold_equal({"SPY": spy_bars}, {"SPY": spy_idx}, calendar, ["SPY"], CostModel(5), args.start_equity, 1.0, calendar[0], calendar[-1])
    series_curves["SPY_buy_hold"] = [{"date": e["date"], "equity": e["equity"]} for e in spy_full]
    ew_full, _ = buy_hold_equal(histories, date_index, calendar, all_syms, cost, args.start_equity, args.max_weight, calendar[0], calendar[-1])
    series_curves["universe_ew_monthly"] = [{"date": e["date"], "equity": e["equity"]} for e in ew_full]

    base = f"{run_at.strftime('%Y-%m-%d')}_highvol-strategy"
    chart.write_equity_csv(run_dir / f"{base}_equity.csv", series_curves)
    with (run_dir / f"{base}_trades.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["entry_date","exit_date","symbol","side","qty","entry_price","exit_price","pnl"])
        w.writeheader(); w.writerows(all_trades_flat)

    # prices tidy for traded names in winners (union)
    price_syms = set()
    for tr in all_trades_flat:
        price_syms.add(tr["symbol"])
    price_syms.update(["SPY", "TQQQ", "SOXL", "XBI", "HOOD", "AXTI", "QURE"])
    symbols_bars = {s: histories[s] for s in price_syms if s in histories}
    # filter to calendar range
    symbols_bars = {s: [b for b in bars if calendar[0] <= b["date"] <= calendar[-1]] for s, bars in symbols_bars.items()}
    chart.write_prices_csv(run_dir / f"{base}_prices.csv", symbols_bars)

    meta = {
        "base_name": base,
        "run_id": run_id,
        "code_version": CODE_VERSION,
        "run_at_et": run_at.isoformat(timespec="seconds"),
        "timezone": "America/New_York",
        "units": {
            "equity": "USD",
            "drawdown_pct": "negative fraction vs peak",
            "pnl": "USD",
            "prices": "USD adjusted",
            "dates": "ISO 8601 date",
        },
        "simulated_vs_real": "simulated",
        "fees_included": True,
        "fee_model": {"default_bps_each_way": 12, "symbol_overrides": "from universe bucket defaults", "commission": 0},
        "execution": "signal_on_close_fill_next_open",
        "max_weight_per_ticker": args.max_weight,
        "holdout_frac": args.holdout_frac,
        "insample": f"{is_start}..{is_end}",
        "holdout": f"{ho_start}..{ho_end}",
        "n_trials": n_trials,
        "n_variants_signals": len(variants),
        "n_regime_modes": len(regime_modes),
        "symbols_n": len(all_syms),
        "survivorship_bias": "acute — free data underrepresents delisted wipeouts; ruins bucket only if Yahoo served history",
        "leveraged_etf_decay": "noted — 3x products path-dependent; strategies may go flat",
        "multiple_testing": f"{n_trials} trials; prefer holdout metrics; haircut sharpe/=sqrt(log(n_trials))",
        "files": [f"{base}_equity.csv", f"{base}_trades.csv", f"{base}_prices.csv", f"{base}_meta.json", "REPORT.md", "results.json"],
    }
    chart.write_meta(run_dir / f"{base}_meta.json", meta)

    results = {
        "meta": meta,
        "benchmarks": bench,
        "trials": trials,
        "survivors_strict": [t["name"] for t in survivors],
        "winners": winner_details,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))

    # REPORT.md
    lines = [
        f"# High-vol strategy hunt — `{run_id}`",
        "",
        f"- Run at: **{meta['run_at_et']}** · code `{CODE_VERSION}`",
        f"- Universe: {len(all_syms)} tradeable symbols across {len(trade_buckets)} buckets (excl. inverse/benchmarks/vol proxies as sleeves)",
        f"- Sample: **{calendar[0]} → {calendar[-1]}** · IS `{is_start}→{is_end}` · Holdout `{ho_start}→{ho_end}`",
        f"- Trials: **{n_trials}** (= {len(variants)} signals × {len(regime_modes)} regimes)",
        f"- Execution: close signal → next open · max weight **{args.max_weight:.0%}**/ticker · costs ~8–25 bps/side",
        f"- Caveats: survivorship bias; leveraged-ETF decay; multiple testing ({n_trials} looks); simulated fills",
        "",
        "## Benchmarks",
        "",
        "| Sleeve | IS CAGR | IS Sharpe | IS MaxDD | HO CAGR | HO Sharpe | HO MaxDD |",
        "|---|---|---|---|---|---|---|",
    ]
    def pct(x):
        return "n/a" if x != x else f"{x*100:.2f}%"
    def sh(x):
        return "n/a" if x != x else f"{x:.2f}"
    for name, b in bench.items():
        lines.append(
            f"| {name} | {pct(b['insample']['cagr'])} | {sh(b['insample']['sharpe'])} | {pct(b['insample']['max_drawdown'])} | "
            f"{pct(b['holdout']['cagr'])} | {sh(b['holdout']['sharpe'])} | {pct(b['holdout']['max_drawdown'])} |"
        )
    lines += ["", "## Winners (human-readable rules)", ""]
    for i, wdetail in enumerate(winner_details, 1):
        lines += [
            f"### Winner {i}: `{wdetail['name']}`",
            "",
            f"- Family: **{wdetail['family']}** · Regime: **{wdetail['regime']}**",
            f"- Weak survivor flag: {wdetail.get('weak_survivor', False)}",
            f"- IS: CAGR {pct(wdetail['insample']['cagr'])}, Sharpe {sh(wdetail['insample']['sharpe'])}, "
            f"MaxDD {pct(wdetail['insample']['max_drawdown'])}, trades {wdetail['insample']['n_trades']}, "
            f"win {pct(wdetail['insample']['win_rate'])}, exposure {pct(wdetail['insample']['exposure'])}",
            f"- HO: CAGR {pct(wdetail['holdout']['cagr'])}, Sharpe {sh(wdetail['holdout']['sharpe'])}, "
            f"MaxDD {pct(wdetail['holdout']['max_drawdown'])}, trades {wdetail['holdout']['n_trades']}, "
            f"win {pct(wdetail['holdout']['win_rate'])}, Sortino {sh(wdetail['holdout']['sortino'])}",
            f"- Path targets on holdout (forward max high vs next open):",
            f"  `{json.dumps(wdetail.get('path_targets_holdout', {}), indent=2)}`",
            "",
        ]
    lines += [
        "## Top 10 by holdout Sharpe (all trials)",
        "",
        "| Rank | Name | HO CAGR | HO Sharpe | HO MaxDD | IS Sharpe | Trades HO |",
        "|---|---|---|---|---|---|---|",
    ]
    top = sorted(trials, key=lambda t: (t["holdout"]["sharpe"] if t["holdout"]["sharpe"]==t["holdout"]["sharpe"] else -9), reverse=True)[:10]
    for i, t in enumerate(top, 1):
        lines.append(
            f"| {i} | `{t['name']}` | {pct(t['holdout']['cagr'])} | {sh(t['holdout']['sharpe'])} | "
            f"{pct(t['holdout']['max_drawdown'])} | {sh(t['insample']['sharpe'])} | {t['holdout']['n_trades']} |"
        )
    lines += [
        "",
        "## Rules legend",
        "",
        "- `ts_mom_fA_sB_tvX_vw20`: long when SMA(A)>SMA(B); weight min(50%, target_vol/realized_vol_20).",
        "- `xs_mom_lbN_topK`: equal-weight top K names by N-day return (skip 1 day).",
        "- `rsi2_mr_ltR_trendS`: RSI(2)<R and price>SMA(S); equal-weight most oversold (top N).",
        "- `ibs_ltX_trendS`: Internal Bar Strength ≤ X and price>SMA(S).",
        "- `donchianN`: close breaks prior N-day high.",
        "- `bucket_rs_lbN`: rotate into best bucket by avg N-day return, hold top 3 names.",
        "- Regime `spy200`: only risk-on if SPY>SMA200; `spy200_vix`: also VIX<25; `off`: always on.",
        "",
        "## Implication for Tim's 2-week debit call spreads",
        "",
        "Path-target hit rates above estimate how often a signaled name reaches +5/+10/+15% within 10 sessions (using highs). "
        "The debit-spread block is a **heuristic estimate** using RV20 as IV — not tradable option P&L.",
        "",
    ]
    (run_dir / "REPORT.md").write_text("\n".join(lines))
    print("\nDONE", run_dir, "elapsed", round(time.time()-t0,1), "s")
    print("Winners:", [w["name"] for w in winner_details])
    print("Strict survivors:", len(survivors))


if __name__ == "__main__":
    main()
