"""
Run SMA crossover vs buy-and-hold on cached daily JSON.

  .venv\Scripts\python.exe -m Backtest.RunBacktest --symbols SPY,HOOD

Walk-forward: train window unused for this simple rule (no fit), but we keep an
untouched holdout (last 20% of bars) and report train/holdout separately.
Never overwrites prior runs: each run gets a stamped out/<run_id>/ folder.
"""
from __future__ import annotations
import argparse, csv, json, hashlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dataclasses import asdict
from Backtest.Engine import CostModel, buy_and_hold, run_long_only, summarize, CODE_VERSION
from Backtest.Strategies.SmaCrossover import make_signal
from Backtest import export_charting as chart

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data" / "daily"
OUT = ROOT / "out"

# Pre-registered primary + secondary variants (log count for multiple-testing caveat)
PRIMARY = (20, 50)
VARIANTS = [(10, 30), (20, 50), (50, 200)]  # 3 tried; primary is 20/50


def load_bars(symbol: str) -> dict:
    path = DAILY / f"{symbol}.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; run DataPull first")
    return json.loads(path.read_text())


def write_run(run_dir: Path, symbol: str, payload: dict, book_strat, book_bh, meta: dict):
    run_dir.mkdir(parents=True, exist_ok=False)
    # trades
    with (run_dir / f"trades_{symbol}_sma.csv").open("w", newline="", encoding="utf-8") as fh:
        rows = [asdict(t) for t in book_strat.trades]
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    with (run_dir / f"equity_{symbol}_sma.csv").open("w", newline="", encoding="utf-8") as fh:
        rows = book_strat.equity_curve
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    with (run_dir / f"equity_{symbol}_bh.csv").open("w", newline="", encoding="utf-8") as fh:
        rows = book_bh.equity_curve
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    (run_dir / f"summary_{symbol}.json").write_text(json.dumps(payload, indent=2))
    return run_dir


def md_report(run_id: str, results: list[dict], meta: dict) -> str:
    lines = [
        f"# Equity SMA backtest — `{run_id}`",
        "",
        f"- **Code version:** {meta['code_version']}",
        f"- **Run at (ET):** {meta['run_at_et']}",
        f"- **Cost model:** commission ${meta['cost']['commission_per_trade']:.2f}/fill, "
        f"slippage {meta['cost']['slippage_bps']} bps + spread {meta['cost']['spread_bps']} bps each way",
        f"- **Execution:** signal on close of bar t → fill at open of bar t+1 (no look-ahead)",
        f"- **Prices:** split-adjusted (Robinhood MCP) or split+div (yfinance auto_adjust); see each symbol's data file",
        f"- **Holdout:** last {meta['holdout_frac']*100:.0f}% of bars untouched for primary rule evaluation",
        f"- **Strategy variants tried this run:** {meta['variants_tried']} (primary={meta['primary']})",
        f"- **Survivorship:** only currently listed symbols; delisted names absent from free sources",
        f"- **Multiple-testing caveat:** {meta['variants_tried']} SMA pairs inspected; holdout numbers for non-primary are exploratory",
        "",
    ]
    for r in results:
        lines += [
            f"## {r['symbol']}",
            "",
            f"- Data source: `{r['data_source']}` · bars {r['bar_count']} · "
            f"{r['first_date']} → {r['last_date']} · fetched_at_et={r['fetched_at_et']} · "
            f"source_last_bar={r['source_last_bar_begins_at']}",
            "",
            "### Primary (SMA 20/50) — in-sample (first 80%)",
            "",
            f"| Metric | SMA 20/50 | Buy & Hold |",
            f"|---|---|---|",
        ]
        s, b = r["insample"]["sma"], r["insample"]["bh"]
        for k in ("cagr", "max_drawdown", "sharpe", "total_return", "n_trades", "win_rate"):
            sv = s.get(k); bv = b.get(k)
            def fmt(v, k=k):
                if v is None or (isinstance(v, float) and v != v):
                    return "n/a"
                if k in ("cagr", "max_drawdown", "total_return", "win_rate", "avg_trade_ret"):
                    return f"{v*100:.2f}%"
                if k == "sharpe":
                    return f"{v:.2f}"
                return str(v)
            lines.append(f"| {k} | {fmt(sv)} | {fmt(bv)} |")
        lines += [
            "",
            "### Primary — holdout (last 20%)",
            "",
            f"| Metric | SMA 20/50 | Buy & Hold |",
            f"|---|---|---|",
        ]
        s, b = r["holdout"]["sma"], r["holdout"]["bh"]
        for k in ("cagr", "max_drawdown", "sharpe", "total_return", "n_trades", "win_rate"):
            sv = s.get(k); bv = b.get(k)
            def fmt(v, k=k):
                if v is None or (isinstance(v, float) and v != v):
                    return "n/a"
                if k in ("cagr", "max_drawdown", "total_return", "win_rate", "avg_trade_ret"):
                    return f"{v*100:.2f}%"
                if k == "sharpe":
                    return f"{v:.2f}"
                return str(v)
            lines.append(f"| {k} | {fmt(sv)} | {fmt(bv)} |")
        lines += ["", "### All variants (full sample, exploratory)", ""]
        lines.append("| Variant | CAGR | MaxDD | Sharpe | Trades | Win% | vs BH CAGR |")
        lines.append("|---|---|---|---|---|---|---|")
        bh_cagr = r["full"]["bh"]["cagr"]
        for name, summ in r["full"]["variants"].items():
            def pct(v):
                return "n/a" if v != v else f"{v*100:.2f}%"
            def sh(v):
                return "n/a" if v != v else f"{v:.2f}"
            edge = summ["cagr"] - bh_cagr if (summ["cagr"] == summ["cagr"] and bh_cagr == bh_cagr) else float("nan")
            lines.append(
                f"| {name} | {pct(summ['cagr'])} | {pct(summ['max_drawdown'])} | {sh(summ['sharpe'])} | "
                f"{summ['n_trades']} | {pct(summ['win_rate'])} | {pct(edge)} |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY,HOOD")
    ap.add_argument("--start-equity", type=float, default=10_000.0)
    ap.add_argument("--holdout-frac", type=float, default=0.20)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--spread-bps", type=float, default=5.0)
    args = ap.parse_args()

    cost = CostModel(slippage_bps=args.slippage_bps, spread_bps=args.spread_bps)
    run_at = datetime.now(ET)
    run_id = f"bt_{run_at.strftime('%Y%m%dT%H%M%S')}_sma"
    run_dir = OUT / run_id
    meta = {
        "run_id": run_id,
        "run_at_et": run_at.isoformat(timespec="seconds"),
        "code_version": CODE_VERSION,
        "cost": {"commission_per_trade": cost.commission_per_trade,
                 "slippage_bps": cost.slippage_bps, "spread_bps": cost.spread_bps},
        "holdout_frac": args.holdout_frac,
        "primary": f"sma_{PRIMARY[0]}_{PRIMARY[1]}",
        "variants_tried": len(VARIANTS),
        "variants": [f"sma_{a}_{b}" for a, b in VARIANTS],
        "execution": "signal_on_close_fill_next_open",
    }
    results = []
    run_dir.mkdir(parents=True, exist_ok=False)

    for sym in [s.strip().upper() for s in args.symbols.split(",") if s.strip()]:
        payload = load_bars(sym)
        bars = payload["bars"]
        n = len(bars)
        cut = int(n * (1.0 - args.holdout_frac))
        # need slow SMA warmup inside window; engine handles via signal returning 0
        primary = make_signal(*PRIMARY)
        # in-sample
        book_is = run_long_only(bars, primary, start_equity=args.start_equity, cost=cost, start_i=0, end_i=cut - 1)
        bh_is = buy_and_hold(bars, start_equity=args.start_equity, cost=cost, start_i=0, end_i=cut - 1)
        # holdout — start flat at cut (no carry of in-sample position)
        book_ho = run_long_only(bars, primary, start_equity=args.start_equity, cost=cost, start_i=cut, end_i=n - 1)
        bh_ho = buy_and_hold(bars, start_equity=args.start_equity, cost=cost, start_i=cut, end_i=n - 1)
        # full sample variants
        variants = {}
        for pair in VARIANTS:
            sig = make_signal(*pair)
            b = run_long_only(bars, sig, start_equity=args.start_equity, cost=cost, start_i=0, end_i=n - 1)
            variants[sig.name] = summarize(b, args.start_equity, n)
        bh_full = buy_and_hold(bars, start_equity=args.start_equity, cost=cost, start_i=0, end_i=n - 1)

        result = {
            "symbol": sym,
            "data_source": payload.get("source"),
            "fetched_at_et": payload.get("fetched_at_et"),
            "source_last_bar_begins_at": payload.get("source_last_bar_begins_at"),
            "adjustment_type": payload.get("adjustment_type"),
            "bar_count": n,
            "first_date": bars[0]["date"],
            "last_date": bars[-1]["date"],
            "insample_bars": cut,
            "holdout_bars": n - cut,
            "insample": {
                "sma": summarize(book_is, args.start_equity, cut),
                "bh": summarize(bh_is, args.start_equity, cut),
            },
            "holdout": {
                "sma": summarize(book_ho, args.start_equity, n - cut),
                "bh": summarize(bh_ho, args.start_equity, n - cut),
            },
            "full": {
                "variants": variants,
                "bh": summarize(bh_full, args.start_equity, n),
            },
        }
        results.append(result)
        # write per-symbol artifacts (primary full-sample equity for charting)
        book_full = run_long_only(bars, primary, start_equity=args.start_equity, cost=cost)
        write_run(run_dir / sym, sym, result, book_full, bh_full, meta)
        print(f"{sym}: IS CAGR sma={result['insample']['sma']['cagr']:.2%} bh={result['insample']['bh']['cagr']:.2%} | "
              f"HO CAGR sma={result['holdout']['sma']['cagr']:.2%} bh={result['holdout']['bh']['cagr']:.2%}")

    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    report = md_report(run_id, results, meta)
    (run_dir / "REPORT.md").write_text(report)

    # Graphtard chart-ready bundle (tidy CSVs, shared base name, never overwrite run_dir)
    base = f"{run_at.strftime('%Y-%m-%d')}_sma-backtest"
    # gather prices + equity + trades across symbols from the just-written per-symbol folders
    symbols_bars = {}
    series_curves = {}
    all_trades = []
    for r in results:
        sym = r["symbol"]
        payload = load_bars(sym)
        symbols_bars[sym] = payload["bars"]
        # re-load equity CSVs we wrote
        eq_sma = list(csv.DictReader((run_dir / sym / f"equity_{sym}_sma.csv").open(encoding="utf-8")))
        eq_bh = list(csv.DictReader((run_dir / sym / f"equity_{sym}_bh.csv").open(encoding="utf-8")))
        series_curves[f"{sym}_sma_20_50"] = [
            {"date": row["date"], "equity": float(row["equity"])} for row in eq_sma
        ]
        series_curves[f"{sym}_buy_hold"] = [
            {"date": row["date"], "equity": float(row["equity"])} for row in eq_bh
        ]
        trades_path = run_dir / sym / f"trades_{sym}_sma.csv"
        if trades_path.exists() and trades_path.stat().st_size > 0:
            for row in csv.DictReader(trades_path.open(encoding="utf-8")):
                all_trades.append({
                    "entry_date": row["entry_date"],
                    "exit_date": row["exit_date"],
                    "symbol": sym,
                    "side": row.get("side", "long"),
                    "qty": float(row["shares"]),
                    "entry_price": float(row["entry_px"]),
                    "exit_price": float(row["exit_px"]),
                    "pnl": float(row["pnl"]),
                })

    chart.write_prices_csv(run_dir / f"{base}_prices.csv", symbols_bars)
    chart.write_equity_csv(run_dir / f"{base}_equity.csv", series_curves)
    # trades helper expects Trade-like; write directly for flat list
    with (run_dir / f"{base}_trades.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "entry_date", "exit_date", "symbol", "side", "qty", "entry_price", "exit_price", "pnl"
        ])
        w.writeheader()
        w.writerows(all_trades)
    chart_meta = {
        "base_name": base,
        "run_id": run_id,
        "code_version": CODE_VERSION,
        "run_at_et": meta["run_at_et"],
        "timezone": "America/New_York",
        "units": {
            "prices": "USD per share, split-adjusted (see adjustment_type per symbol data file)",
            "equity": "USD account equity starting at start_equity",
            "drawdown_pct": "negative fraction vs running peak (e.g. -0.12 = -12%)",
            "pnl": "USD",
            "qty": "shares",
            "dates": "ISO 8601 calendar date (daily bars); intraday would include offset",
        },
        "simulated_vs_real": "simulated",
        "fees_included": True,
        "fee_model": meta["cost"],
        "execution": meta["execution"],
        "holdout_frac": meta["holdout_frac"],
        "variants_tried": meta["variants_tried"],
        "primary": meta["primary"],
        "symbols": [r["symbol"] for r in results],
        "data_sources": {r["symbol"]: r["data_source"] for r in results},
        "files": [
            f"{base}_prices.csv",
            f"{base}_equity.csv",
            f"{base}_trades.csv",
            f"{base}_meta.json",
            "REPORT.md",
            "results.json",
        ],
        "survivorship_bias": "only currently listed symbols from free/RH sources; delistings absent",
        "notes": "Plain numbers only in CSVs. Drawdown is a fraction, not percent points.",
    }
    chart.write_meta(run_dir / f"{base}_meta.json", chart_meta)
    # also copy tidy prices snapshot under out/ for standalone charting of the cache
    prices_base = f"{run_at.strftime('%Y-%m-%d')}_daily-prices"
    prices_dir = OUT / f"prices_{run_at.strftime('%Y%m%dT%H%M%S')}"
    prices_dir.mkdir(parents=True, exist_ok=False)
    chart.write_prices_csv(prices_dir / f"{prices_base}.csv", symbols_bars)
    chart.write_meta(prices_dir / f"{prices_base}_meta.json", {
        "base_name": prices_base,
        "timezone": "America/New_York",
        "units": {"prices": "USD", "volume": "shares", "dates": "ISO 8601 date"},
        "simulated_vs_real": "real historical bars from source",
        "fees_included": False,
        "symbols": list(symbols_bars),
        "data_sources": {r["symbol"]: r["data_source"] for r in results},
        "bar_counts": {r["symbol"]: r["bar_count"] for r in results},
    })

    print(f"\nWrote {run_dir}")
    print(report)


if __name__ == "__main__":
    main()
