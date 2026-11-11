# stonk-data

Keyless (no API key) equity data + leak-free daily backtests for Optimus-Stonk.

## Layout (mirrors sports bots)

- `DataPull/` — fetch + cache only (yfinance primary; Robinhood MCP also used offline)
- `data/daily/<SYM>.json` — canonical daily OHLCV (readable JSON)
- `cache/raw/` — stamped raw pulls (never delete casually)
- `Backtest/` — Engine, Strategies/*, RunBacktest, PredictDay
- `out/<run_id>/` — stamped outputs; never overwritten

## Commands

```bash
cd /workspace/stonk-data
.venv\Scripts\python.exe -m DataPull.yfinance_daily --symbols SPY,HOOD,AXTI,SHOP,QURE
.venv\Scripts\python.exe -m Backtest.RunBacktest --symbols SPY,HOOD
.venv\Scripts\python.exe -m Backtest.PredictDay --symbols SPY,HOOD
.venv\Scripts\python.exe -m DataPull.quotes --symbols SPY,HOOD   # exit 2 if stale
```

## Conventions

- Signal on close of bar t → execute at open of t+1
- Costs: slippage + spread bps
- Holdout last 20%; log variant count
- Source timestamp ≠ fetch time; fail closed on stale live quotes
- Adjusted prices; survivorship bias noted

## 2026-11-09 additions (on Tomputer; Windows: `.venv\Scripts\python.exe`)

- `DataPull/market_calendar.py`: NYSE sessions/holidays, last *completed* session (16:30 ET rule).
- `DataPull/refresh.py`: incremental universe refresh, partial-session drop, re-adjust detection, health report `data/health/latest.md`, **exit 2 = fail closed**. (Written, not yet run.)
- `DataPull/earnings.py`: keyless yfinance earnings dates â†’ `data/earnings/` (pulled for HOOD, AXTI, SHOP, QURE).
- `DataPull/snapshot.py`: delayed-quote snapshot CSV + 6-month draft PNG.
- `Backtest/VecMomentum.py`, `Backtest/RobustSweep.py`: fast momentum engine + 576-config sweep/walk-forward. (Written, not yet run.) Finding: the old `mt_` rule's IS CAGR ranges 36â€“81% depending only on the rebalance weekday (phase).
- `Options/`: synthetic options engine (MODEL prices, not quotes): `Pricing` (BSM + Greeks, CRR American, bounds, payoffs), `VolModel` (trailing vol, labeled IV scenarios), `Structures`, `DayTradeGuard` (â‰¤3 per 5 sessions), `SynthBacktest`, `Report`, `Calibrate`.
- Tests: `.venv\Scripts\python -m unittest tests.test_options -v` (19 tests).
- Run: `.venv\Scripts\python -m Options.SynthBacktest --procs 22` (~40 s; writes `out/so_<stamp>/`). Smoke: `--quick`.
- Calibration: `.venv\Scripts\python -m Options.Calibrate --out out\calib.csv` (uses `data/calibration/rh_calib_*.json` from read-only Robinhood quotes).


## 2026-11-09 options validation (run out/ov_20261109T155334, appended; earlier sections unchanged)
- New: DataPull/raw_daily.py (unadjusted prices + splits/dividends -> data/daily_raw), Options/RealHistory.py (+RealHistoryBuckets.py, RealHistoryVrp.py) real RH option panel check, Options/Validate10k.py (v2 engine: raw strikes, split exclusion, dividend escrow/credit, real min ticks, 1-5% sizing grid, exposure/assignment flags), tests/test_validate.py.
- Real panel: data/calibration/rh_option_daily_20261109.json (46 expired + 2 active contracts, 6 tickers, daily bars, read-only).
- Result: no option variant robust at $10k; premium-selling edge depends on assumed IVRP; shares momentum all 10 phases positive but below SPY buy&hold on pre-holdout Sharpe; nested OOS 12.5% CAGR vs SPY 13.2%.
- Deliverables: E:\AgentDocs\Optimus-Stonk\2026-11-09_options-validation\ (methodology .md first).
- Tests: .venv\Scripts\python -m unittest tests.test_validate tests.test_options -v (28 tests).

## 2026-11-09 event phase (run out/ev_20261109T160721, appended; earlier sections unchanged)
- New: Events/Hypotheses.py (pre-registration, sha256 dcdefff6...), Events/Hypotheses.py.deviations.md, Events/Registry.py (FOMC from federalreserve.gov, CPI from BLS archive via Wayback copy, earnings timing check) -> data/events/registry.csv, Events/EventStudy.py, tests/test_events.py; data/events_cpi_bls.csv, data/bls_cpi_archive_wayback_20261101.html, data/calibration/rh_option_daily_20261109_v2.json (+QQQ/TLT/GDX/XBI).
- Result: no event rule robust after placebo + Holm; earnings variance real, direction not predictable; macro-event avoidance hurts momentum (~-10%/yr), earnings avoidance ~ -2.7%/yr CAGR, Sharpe ~flat. Found ITM half-spread open-exit artifact in Validate10k fills (affects tminus1 rows).
- Deliverables: E:\AgentDocs\Optimus-Stonk\2026-11-09_event-options\ (results .md first). Tests: .venv\Scripts\python -m unittest tests.test_events tests.test_validate tests.test_options -v (34).
