"""
Make XS-momentum tradeable for a ~$115 account: fix overlays, early OOS,
DD-controlled variants, options translation, Monte Carlo to $1M.
"""
from __future__ import annotations
import csv, json, math, random, time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from Backtest.MultiAssetEngine import Portfolio, CostModel, summarize, Trade, CODE_VERSION
from Backtest.Strategies.HighVolFamilies import sma, realized_vol, atr
from Backtest import export_charting as chart
from Backtest.HuntHighVol import load_bars, index_by_date, buy_hold_equal
from Backtest.CashOverlayHunt import precompute_features, OverlayState

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
UNIVERSE = ROOT / "data/universe/highvol_2026-11-08.json"
ACCOUNT0 = 115.0


# ---------------- signals / variants ----------------
def xs_signal(histories, i_by_sym, lookback, top_n, skip=1, inv_vol=False, max_w=0.50,
              buckets=None, max_per_bucket=99, atr_stop_mult=None, stop_state=None):
    """Cross-sectional momentum with optional inv-vol weights and bucket caps.
    atr_stop_mult: if set with stop_state dict, zero weight names that hit -atr trail from entry high.
    """
    scores = []
    for sym, bars in histories.items():
        i = i_by_sym.get(sym)
        if i is None or i < lookback + skip:
            continue
        # trailing stop check (uses prior entry tracking)
        if stop_state is not None and atr_stop_mult is not None and sym in stop_state:
            st = stop_state[sym]
            a = atr(bars[: i + 1], 14)
            if a and bars[i]["close"] < st["peak"] - atr_stop_mult * a:
                continue  # stopped out — skip from new picks today
            # also -15% from peak option encoded as atr_stop_mult=None with pct in stop_state
        if stop_state is not None and stop_state.get(sym, {}).get("pct_stop"):
            st = stop_state[sym]
            if bars[i]["close"] < st["peak"] * (1 - st["pct_stop"]):
                continue
        c_now = bars[i - skip]["close"]
        c_past = bars[i - skip - lookback]["close"]
        if c_past <= 0:
            continue
        ret = c_now / c_past - 1
        if ret <= 0:
            continue
        vol = realized_vol([b["close"] for b in bars[: i + 1]], 20) or 0.5
        scores.append((sym, ret, vol))
    scores.sort(key=lambda x: x[1], reverse=True)

    picks = []
    bucket_count = {}
    for sym, ret, vol in scores:
        if len(picks) >= top_n:
            break
        if buckets and max_per_bucket < 99:
            b = buckets.get(sym, "other")
            if bucket_count.get(b, 0) >= max_per_bucket:
                continue
            bucket_count[b] = bucket_count.get(b, 0) + 1
        picks.append((sym, ret, vol))
    if not picks:
        return {}
    if inv_vol:
        inv = [(s, 1.0 / max(v, 0.05)) for s, _, v in picks]
        ssum = sum(x[1] for x in inv)
        w = {s: min(max_w, x / ssum) for s, x in inv}
    else:
        wt = min(max_w, 1.0 / len(picks))
        w = {s: wt for s, _, _ in picks}
    ssum = sum(w.values())
    if ssum > 1.0:
        w = {k: v / ssum for k, v in w.items()}
    return w


def ts_signal(histories, i_by_sym, fast=20, slow=100, target_vol=0.20, max_w=0.50):
    w = {}
    for sym, bars in histories.items():
        i = i_by_sym.get(sym)
        if i is None or i < slow:
            continue
        closes = [b["close"] for b in bars[: i + 1]]
        sf, ss = sma(closes, fast), sma(closes, slow)
        if sf is None or ss is None or sf <= ss:
            continue
        vol = realized_vol(closes, 20) or target_vol
        w[sym] = min(max_w, target_vol / max(vol, 1e-6))
    s = sum(w.values())
    if s > 1.0:
        w = {k: v / s for k, v in w.items()}
    return w


VARIANT_SPECS = [
    # name, lookback, top_n, inv_vol, rebalance_days, vol_target, max_per_bucket, pct_stop, atr_stop
    {"name": "xs126_top3_ew_daily", "lb": 126, "top": 3, "inv_vol": False, "reb": 1, "pvol": None, "mpb": 99, "pct_stop": None, "atr_stop": None},
    {"name": "xs126_top5_ew_daily", "lb": 126, "top": 5, "inv_vol": False, "reb": 1, "pvol": None, "mpb": 99, "pct_stop": None, "atr_stop": None},
    {"name": "xs126_top8_ew_daily", "lb": 126, "top": 8, "inv_vol": False, "reb": 1, "pvol": None, "mpb": 99, "pct_stop": None, "atr_stop": None},
    {"name": "xs126_top5_invvol_daily", "lb": 126, "top": 5, "inv_vol": True, "reb": 1, "pvol": None, "mpb": 99, "pct_stop": None, "atr_stop": None},
    {"name": "xs126_top5_ew_weekly", "lb": 126, "top": 5, "inv_vol": False, "reb": 5, "pvol": None, "mpb": 99, "pct_stop": None, "atr_stop": None},
    {"name": "xs126_top5_ew_biweekly", "lb": 126, "top": 5, "inv_vol": False, "reb": 10, "pvol": None, "mpb": 99, "pct_stop": None, "atr_stop": None},
    {"name": "xs126_top5_pvol40_weekly", "lb": 126, "top": 5, "inv_vol": False, "reb": 5, "pvol": 0.40, "mpb": 99, "pct_stop": None, "atr_stop": None},
    {"name": "xs126_top5_pvol60_weekly", "lb": 126, "top": 5, "inv_vol": False, "reb": 5, "pvol": 0.60, "mpb": 99, "pct_stop": None, "atr_stop": None},
    {"name": "xs126_top5_bucket2_weekly", "lb": 126, "top": 5, "inv_vol": False, "reb": 5, "pvol": None, "mpb": 2, "pct_stop": None, "atr_stop": None},
    {"name": "xs126_top5_pctstop15_weekly", "lb": 126, "top": 5, "inv_vol": False, "reb": 5, "pvol": None, "mpb": 99, "pct_stop": 0.15, "atr_stop": None},
    {"name": "xs63_top5_pvol40_weekly", "lb": 63, "top": 5, "inv_vol": False, "reb": 5, "pvol": 0.40, "mpb": 99, "pct_stop": None, "atr_stop": None},
    {"name": "xs63_top3_ew_weekly", "lb": 63, "top": 3, "inv_vol": False, "reb": 5, "pvol": None, "mpb": 99, "pct_stop": None, "atr_stop": None},
]


def scale_to_pvol(weights, histories, i_by_sym, target_pvol):
    """Scale all weights by target / recent portfolio vol estimate (diag cov approx via avg)."""
    if not weights or not target_pvol:
        return weights
    vols = []
    for sym, w in weights.items():
        bars = histories[sym]
        i = i_by_sym[sym]
        v = realized_vol([b["close"] for b in bars[: i + 1]], 20)
        if v:
            vols.append((w, v))
    if not vols:
        return weights
    # crude port vol ~ sqrt(sum (w*v)^2) assuming low corr for high-vol names — use 0.5 corr blend
    # better: equal-corr 0.4
    corr = 0.4
    var = 0.0
    for i, (wi, vi) in enumerate(vols):
        var += (wi * vi) ** 2
        for j, (wj, vj) in enumerate(vols):
            if j > i:
                var += 2 * corr * wi * wj * vi * vj
    pvol = math.sqrt(max(var, 1e-12))
    scale = min(2.0, target_pvol / pvol)  # cap leverage
    out = {s: min(0.50, w * scale) for s, w in weights.items()}
    ssum = sum(out.values())
    if ssum > 1.0:
        out = {k: v / ssum for k, v in out.items()}
    return out


@dataclass
class RunResult:
    curve: list
    trades: list
    daily_rets: list
    summary: dict
    last_weights: dict = field(default_factory=dict)


def run_variant(histories, date_index, calendar, spec, cost, start_equity, buckets_map,
                start_date, end_date, shadow_for_stops=True):
    dates = [d for d in calendar if start_date <= d <= end_date]
    port = Portfolio(start_equity, cost, max_weight=0.50)
    pending = None
    stop_state = {}  # sym -> {peak, pct_stop}
    last_reb_i = -10**9
    last_w = {}
    for di, date in enumerate(dates):
        opens, closes = {}, {}
        i_by = {}
        for sym, bars in histories.items():
            j = date_index[sym].get(date)
            if j is None:
                continue
            i_by[sym] = j
            opens[sym] = bars[j]["open"]
            closes[sym] = bars[j]["close"]
            # update peaks for stops
            if sym in port.positions:
                st = stop_state.setdefault(sym, {"peak": closes[sym], "pct_stop": spec["pct_stop"]})
                st["peak"] = max(st["peak"], closes[sym])
                if spec["pct_stop"]:
                    st["pct_stop"] = spec["pct_stop"]

        if pending is not None:
            port.execute_targets(date, opens, pending)
            pending = None
            # refresh stop peaks on new entries
            for sym in port.positions:
                stop_state[sym] = {"peak": opens.get(sym, port.positions[sym]["entry_px"]),
                                   "pct_stop": spec["pct_stop"]}

        port.mark(date, closes if closes else {s: port.positions[s]["entry_px"] for s in port.positions})

        # rebalance?
        do_reb = (di - last_reb_i) >= spec["reb"]
        if do_reb:
            last_reb_i = di
            # apply pct stops: drop names that breached
            active_stop = {}
            if spec["pct_stop"]:
                for sym, st in stop_state.items():
                    active_stop[sym] = st
            w = xs_signal(
                histories, i_by, spec["lb"], spec["top"], skip=1,
                inv_vol=spec["inv_vol"], max_w=0.50, buckets=buckets_map,
                max_per_bucket=spec["mpb"], stop_state=active_stop if spec["pct_stop"] else None,
            )
            if spec["pvol"]:
                w = scale_to_pvol(w, histories, i_by, spec["pvol"])
            pending = w
            last_w = dict(w)
        else:
            # hold prior targets conceptually — only exit via stops mid-cycle
            if spec["pct_stop"] and port.positions:
                # force flatten stopped names next open
                new_targets = {s: port.positions[s]["qty"] * closes.get(s, 0) for s in port.positions}
                # convert to weights roughly at mark
                eq = port.curve[-1]["equity"]
                w = {}
                for sym in list(port.positions.keys()):
                    st = stop_state.get(sym)
                    if st and st.get("pct_stop") and closes.get(sym, 0) < st["peak"] * (1 - st["pct_stop"]):
                        continue  # drop
                    if eq > 0 and sym in closes:
                        w[sym] = min(0.50, port.positions[sym]["qty"] * closes[sym] / eq)
                # only update pending if someone stopped
                if set(w.keys()) != set(port.positions.keys()):
                    pending = w
                    last_w = dict(w)

    eqs = [e["equity"] for e in port.curve]
    rets = []
    for i in range(1, len(eqs)):
        if eqs[i - 1] > 0:
            rets.append(eqs[i] / eqs[i - 1] - 1)
    summ = summarize(port.curve, port.trades, start_equity, len(dates))
    # turnover approx already in summarize
    return RunResult(port.curve, port.trades, rets, summ, last_w)


# -------- fixed strat_dd overlay using shadow equity --------
def run_with_fixed_strat_dd(histories, date_index, calendar, signal_fn, cost, start_equity,
                            start_date, end_date, reenter_mode="shadow_recover_6"):
    """
    Fix: maintain shadow equity of always-on strategy; go cash when LIVE dd<=-12%;
    re-enter when SHADOW dd recovers to -6% OR (if mode includes regime) SPY>SMA50.
    """
    dates = [d for d in calendar if start_date <= d <= end_date]
    live = Portfolio(start_equity, cost, max_weight=0.50)
    shadow = Portfolio(start_equity, cost, max_weight=0.50)
    pending_live = None
    pending_shadow = None
    in_cash = False
    live_peak = start_equity
    switches = 0
    prev_cash = False
    log = []

    # need SPY for optional regime reentry
    spy = load_bars("SPY")
    spy_idx = index_by_date(spy)

    for date in dates:
        opens, closes, i_by = {}, {}, {}
        for sym, bars in histories.items():
            j = date_index[sym].get(date)
            if j is None:
                continue
            i_by[sym] = j
            opens[sym] = bars[j]["open"]
            closes[sym] = bars[j]["close"]

        if pending_shadow is not None:
            shadow.execute_targets(date, opens, pending_shadow)
            pending_shadow = None
        if pending_live is not None:
            live.execute_targets(date, opens, pending_live)
            pending_live = None

        shadow.mark(date, closes if closes else {})
        live.mark(date, closes if closes else {})
        sh_eq = shadow.curve[-1]["equity"]
        lv_eq = live.curve[-1]["equity"]
        live_peak = max(live_peak, lv_eq)
        live_dd = lv_eq / live_peak - 1.0 if live_peak > 0 else 0.0
        # shadow peak for recovery signal
        # use shadow's own peak from its curve
        sh_eqs = [e["equity"] for e in shadow.curve]
        sh_peak = max(sh_eqs) if sh_eqs else start_equity
        sh_dd = sh_eq / sh_peak - 1.0 if sh_peak > 0 else 0.0

        spy_i = spy_idx.get(date)
        spy_ok = False
        if spy_i is not None and spy_i >= 50:
            sc = [spy[k]["close"] for k in range(spy_i + 1)]
            s50 = sma(sc, 50)
            spy_ok = s50 is not None and sc[-1] > s50

        if not in_cash:
            if live_dd <= -0.12:
                in_cash = True
        else:
            if reenter_mode == "shadow_recover_6" and sh_dd >= -0.06:
                in_cash = False
                live_peak = lv_eq  # reset peak on re-entry
            elif reenter_mode == "shadow_or_spy50" and (sh_dd >= -0.06 or spy_ok):
                in_cash = False
                live_peak = lv_eq
            elif reenter_mode == "spy50_only" and spy_ok:
                in_cash = False
                live_peak = lv_eq

        if in_cash != prev_cash:
            switches += 1
            prev_cash = in_cash

        w = signal_fn(histories, i_by, True)
        pending_shadow = w
        pending_live = {} if in_cash else w
        log.append({"date": date, "in_cash": int(in_cash), "live_dd": live_dd, "shadow_dd": sh_dd})

    cash_pct = sum(x["in_cash"] for x in log) / len(log) if log else 0
    return live.curve, live.trades, log, {"pct_time_cash": cash_pct, "n_switches": switches}


# -------- Black-Scholes helpers --------
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_call(S, K, T, sigma, r=0.04):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def simulate_debit_spreads(picks_over_time, histories, date_index, account0=ACCOUNT0,
                           iv_mult=1.2, width=0.10, haircut=0.08, hold_days=10,
                           long_mode="atm"):
    """
    picks_over_time: list of {date, weights}
    ESTIMATED BS debit spreads — not real option prices.
    """
    equity = account0
    curve = []
    trades = []
    cash = account0
    # simple: each rebalance date, close prior synthetic, open new
    open_pos = []  # {sym, long_k, short_k, debit, n_spreads, entry_date, entry_i, iv}

    for item in picks_over_time:
        date = item["date"]
        weights = item["weights"]
        # mark / close positions that reached hold_days
        still = []
        for pos in open_pos:
            bars = histories[pos["sym"]]
            j = date_index[pos["sym"]].get(date)
            if j is None:
                still.append(pos)
                continue
            held = j - pos["entry_i"]
            if held >= hold_days or pos["sym"] not in weights:
                S = bars[j]["close"]
                T = max(0.0, (hold_days - held) / 252.0)
                # exit mid with haircut against us
                long_v = bs_call(S, pos["long_k"], T, pos["iv"])
                short_v = bs_call(S, pos["short_k"], T, pos["iv"])
                mid = max(0.0, long_v - short_v)
                exit_px = mid * (1 - haircut)
                pnl = (exit_px - pos["debit"]) * 100 * pos["n_spreads"]
                cash += exit_px * 100 * pos["n_spreads"]
                trades.append({
                    "entry_date": pos["entry_date"], "exit_date": date, "symbol": pos["sym"],
                    "side": "debit_call_spread", "qty": pos["n_spreads"],
                    "entry_price": pos["debit"], "exit_price": exit_px, "pnl": pnl,
                })
            else:
                still.append(pos)
        open_pos = still

        # open new for weights not already held
        held_syms = {p["sym"] for p in open_pos}
        for sym, w in weights.items():
            if sym in held_syms or w <= 0:
                continue
            bars = histories[sym]
            j = date_index[sym].get(date)
            if j is None or j < 21:
                continue
            # enter next bar open if possible
            if j + 1 >= len(bars):
                continue
            entry_i = j + 1
            S = bars[entry_i]["open"]
            closes = [b["close"] for b in bars[: j + 1]]
            rv = realized_vol(closes, 20) or 0.5
            iv = max(0.15, min(2.0, rv * iv_mult))
            # path drift: 10d trailing
            if len(closes) >= 11 and closes[-11] > 0:
                drift = closes[-1] / closes[-11] - 1
            else:
                drift = 0.0
            if long_mode == "atm":
                long_k = round(S)
            elif long_mode == "plus5":
                long_k = round(S * 1.05)
            else:  # path
                long_k = round(S * (1 + max(0.0, drift)))
            short_k = round(long_k * (1 + width))
            if short_k <= long_k:
                short_k = long_k + max(1, round(S * width))
            T = hold_days / 252.0
            debit_mid = max(0.05, bs_call(S, long_k, T, iv) - bs_call(S, short_k, T, iv))
            debit = debit_mid * (1 + haircut)  # pay through spread
            # size: <=50% of equity in this ticker, whole contracts
            # size on cash only (conservative; open MTM not counted as buying power)
            budget = min(0.50 * max(cash, account0), cash)
            cost_1 = debit * 100
            if cost_1 <= 0 or budget < cost_1:
                continue  # cannot afford even 1 spread
            n = int(budget // cost_1)
            if n < 1:
                continue
            cash -= n * cost_1
            open_pos.append({
                "sym": sym, "long_k": float(long_k), "short_k": float(short_k),
                "debit": debit, "n_spreads": n, "entry_date": bars[entry_i]["date"],
                "entry_i": entry_i, "iv": iv,
            })

        # mark equity
        mtm = cash
        for p in open_pos:
            j = date_index[p["sym"]].get(date)
            if j is None:
                continue
            S = histories[p["sym"]][j]["close"]
            held = j - p["entry_i"]
            T = max(0.0, (hold_days - held) / 252.0)
            mid = max(0.0, bs_call(S, p["long_k"], T, p["iv"]) - bs_call(S, p["short_k"], T, p["iv"]))
            mtm += mid * (1 - haircut) * 100 * p["n_spreads"]
        curve.append({"date": date, "equity": mtm})

    return curve, trades


# -------- Monte Carlo --------
def block_bootstrap_paths(daily_rets, n_paths, n_days, block=5, seed=42):
    rng = random.Random(seed)
    if len(daily_rets) < block + 1:
        return []
    paths = []
    for _ in range(n_paths):
        series = []
        while len(series) < n_days:
            start = rng.randrange(0, len(daily_rets) - block)
            series.extend(daily_rets[start:start + block])
        paths.append(series[:n_days])
    return paths


def millionaire_mc(daily_rets, start=ACCOUNT0, deposits_monthly=0.0, haircut_edge=0.5,
                   n_paths=2000, years=(1, 3, 5, 10), seed=7):
    """haircut_edge: multiply mean excess return toward 0 by this factor on each day? 
    Simpler: scale each return r -> haircut_edge * r (halve edge AND vol — conservative).
    Alternate: r_adj = r - (1-haircut)*mean(r) to shrink mean only — use scale for main case.
    """
    if not daily_rets:
        return {}
    mean_r = sum(daily_rets) / len(daily_rets)
    # main: shrink toward zero: r' = haircut_edge * r
    adj = [haircut_edge * r for r in daily_rets]
    out = {"raw": {}, "haircut": {}, "params": {
        "start": start, "deposits_monthly": deposits_monthly, "haircut_edge": haircut_edge,
        "n_paths": n_paths, "block": 5, "note": "haircut scales each daily return by haircut_edge",
    }}
    targets = [1_000, 10_000, 100_000, 1_000_000]
    for label, rets in [("raw", daily_rets), ("haircut", adj)]:
        out[label] = {}
        for y in years:
            n_days = int(y * 252)
            paths = block_bootstrap_paths(rets, n_paths, n_days, block=5, seed=seed + y)
            hit = {t: 0 for t in targets}
            ruin80 = 0
            finals = []
            # monthly deposit every 21 trading days
            for path in paths:
                eq = start
                peak = eq
                hit_t = {t: False for t in targets}
                ruined = False
                for i, r in enumerate(path):
                    eq *= (1 + r)
                    if deposits_monthly and i > 0 and i % 21 == 0:
                        eq += deposits_monthly
                    peak = max(peak, eq)
                    if peak > 0 and eq / peak - 1.0 <= -0.80:
                        ruined = True
                    for t in targets:
                        if eq >= t:
                            hit_t[t] = True
                for t in targets:
                    if hit_t[t]:
                        hit[t] += 1
                if ruined:
                    ruin80 += 1
                finals.append(eq)
            finals.sort()
            def pctile(p):
                return finals[int((len(finals) - 1) * p)]
            out[label][str(y)] = {
                "prob_1k": hit[1000] / n_paths,
                "prob_10k": hit[10000] / n_paths,
                "prob_100k": hit[100000] / n_paths,
                "prob_1M": hit[1000000] / n_paths,
                "prob_dd80": ruin80 / n_paths,
                "p10_final": pctile(0.10),
                "p50_final": pctile(0.50),
                "p90_final": pctile(0.90),
            }
    return out


def main():
    t0 = time.time()
    u = json.loads(UNIVERSE.read_text())
    exclude = {"leveraged_inverse", "benchmarks", "vol_proxy"}
    buckets_map = {}
    cost_bps = {}
    all_syms = []
    for bname, b in u["buckets"].items():
        if bname in exclude:
            continue
        for s in b["symbols"]:
            bars = load_bars(s)
            if bars and len(bars) >= 100:
                all_syms.append(s)
                buckets_map[s] = bname
                cost_bps[s] = b.get("default_cost_bps", 12)
    all_syms = sorted(set(all_syms))
    histories = {s: load_bars(s) for s in all_syms}
    date_index = {s: index_by_date(histories[s]) for s in all_syms}
    spy = load_bars("SPY"); spy_idx = index_by_date(spy)
    cost = CostModel(default_bps=12.0, symbol_bps=cost_bps)

    # calendars
    cal_full = [b["date"] for b in spy if "2010-01-01" <= b["date"] <= "2026-11-06"]
    cal_early = [d for d in cal_full if d <= "2017-12-31"]
    cal_is = [d for d in cal_full if "2018-01-01" <= d <= "2025-01-30"]
    cal_ho = [d for d in cal_full if "2025-01-31" <= d <= "2026-11-06"]
    # early universe: names with data from 2010
    early_syms = [s for s in all_syms if histories[s][0]["date"] <= "2010-06-01" and len(histories[s]) >= 500]
    early_hist = {s: histories[s] for s in early_syms}
    early_idx = {s: date_index[s] for s in early_syms}
    print(f"Symbols full={len(all_syms)} early2010={len(early_syms)}")

    run_at = datetime.now(ET)
    run_id = f"mt_{run_at.strftime('%Y%m%dT%H%M%S')}"
    run_dir = OUT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    base = f"{run_at.strftime('%Y-%m-%d')}_momentum-tradeable"

    results = {"run_id": run_id, "account0": ACCOUNT0, "trials": [], "notes": []}

    # ===== 1) Fix strat_dd =====
    print("=== Fix strat_dd overlay ===", flush=True)
    def sig126(h, i_by, regime_ok):
        return xs_signal(h, i_by, 126, 3) if regime_ok else {}
    dd_fix = {}
    for mode in ["shadow_recover_6", "shadow_or_spy50", "spy50_only"]:
        for window, s, e in [("IS", cal_is[0], cal_is[-1]), ("HO", cal_ho[0], cal_ho[-1]), ("EARLY", cal_early[0], cal_early[-1])]:
            hist = early_hist if window == "EARLY" else histories
            didx = early_idx if window == "EARLY" else date_index
            cal = cal_early if window == "EARLY" else (cal_is if window == "IS" else cal_ho)
            curve, trades, log, extra = run_with_fixed_strat_dd(
                hist, didx, cal, sig126, cost, 10_000.0, s, e, reenter_mode=mode,
            )
            summ = summarize(curve, trades, 10_000.0, len(cal))
            summ.update(extra)
            dd_fix.setdefault(mode, {})[window] = summ
            print(f"  {mode} {window}: CAGR={summ['cagr']*100:.1f}% Sh={summ['sharpe']:.2f} DD={summ['max_drawdown']*100:.1f}% cash={summ['pct_time_cash']*100:.0f}% sw={summ['n_switches']}")
    results["fixed_strat_dd"] = dd_fix

    # ===== 2) Frozen rules early OOS + rolling WF =====
    print("=== Frozen rules early OOS ===", flush=True)
    frozen = {
        "xs126_top3": lambda h, i, r: xs_signal(h, i, 126, 3),
        "xs63_top3": lambda h, i, r: xs_signal(h, i, 63, 3),
        "ts_mom": lambda h, i, r: ts_signal(h, i),
    }
    frozen_res = {}
    for name, fn in frozen.items():
        frozen_res[name] = {}
        for label, cal, hist, didx in [
            ("EARLY_2010_2017", cal_early, early_hist, early_idx),
            ("IS_2018_2025", cal_is, histories, date_index),
            ("HO_2025_2026", cal_ho, histories, date_index),
        ]:
            # wrap as variant runner with daily top3
            spec = {"name": name, "lb": 126 if "126" in name else 63, "top": 3, "inv_vol": False,
                    "reb": 1, "pvol": None, "mpb": 99, "pct_stop": None, "atr_stop": None}
            if name == "ts_mom":
                # custom loop
                port = Portfolio(10_000.0, cost, 0.50)
                pending = None
                for date in cal:
                    opens, closes, i_by = {}, {}, {}
                    for sym, bars in hist.items():
                        j = didx[sym].get(date)
                        if j is None: continue
                        i_by[sym] = j; opens[sym]=bars[j]["open"]; closes[sym]=bars[j]["close"]
                    if pending is not None:
                        port.execute_targets(date, opens, pending); pending=None
                    port.mark(date, closes if closes else {})
                    pending = ts_signal(hist, i_by)
                summ = summarize(port.curve, port.trades, 10_000.0, len(cal))
                rets = []
                eqs=[e["equity"] for e in port.curve]
                for i in range(1,len(eqs)):
                    if eqs[i-1]>0: rets.append(eqs[i]/eqs[i-1]-1)
            else:
                rr = run_variant(hist, didx, cal, spec, cost, 10_000.0, buckets_map, cal[0], cal[-1])
                summ, rets = rr.summary, rr.daily_rets
            frozen_res[name][label] = summ
            print(f"  {name} {label}: CAGR={summ['cagr']*100:.1f}% Sh={summ['sharpe']:.2f} DD={summ['max_drawdown']*100:.1f}%")
    results["frozen_period_tests"] = frozen_res

    # Rolling-origin WF: train 3y expand? Use origins every year 2013..2022 test next year
    print("=== Rolling-origin walk-forward ===", flush=True)
    rolling = []
    for year in range(2013, 2026):
        train_end = f"{year}-01-01"
        test_start = f"{year}-01-01"
        test_end = f"{year}-12-31"
        if year == 2026:
            test_end = "2026-11-06"
        test_cal = [d for d in cal_full if test_start <= d <= test_end]
        if len(test_cal) < 50:
            continue
        # use early+full hist as available
        spec = VARIANT_SPECS[0]  # xs126 top3 daily frozen
        # symbols that existed 1y before test
        hist = {s: histories[s] for s in all_syms if histories[s][0]["date"] <= f"{year-2}-01-01"}
        didx = {s: date_index[s] for s in hist}
        rr = run_variant(hist, didx, test_cal, spec, cost, 10_000.0, buckets_map, test_cal[0], test_cal[-1])
        rolling.append({"test_year": year, "n_syms": len(hist), **{k: rr.summary[k] for k in
                        ("cagr","sharpe","max_drawdown","n_trades","total_return")}})
        print(f"  WF {year}: CAGR={rr.summary['cagr']*100:.1f}% Sh={rr.summary['sharpe']:.2f} DD={rr.summary['max_drawdown']*100:.1f}% n={len(hist)}")
    results["rolling_wf_xs126_top3"] = rolling

    # ===== 3) DD-control variant hunt =====
    print("=== Variant hunt (pick on IS) ===", flush=True)
    variant_rows = []
    for spec in VARIANT_SPECS:
        print(f"  {spec['name']}...", flush=True)
        row = {"name": spec["name"], "spec": spec}
        for label, cal in [("IS", cal_is), ("HO", cal_ho), ("EARLY", cal_early)]:
            hist = early_hist if label == "EARLY" else histories
            didx = early_idx if label == "EARLY" else date_index
            # for early, filter bucket map
            rr = run_variant(hist, didx, cal, spec, cost, 10_000.0, buckets_map, cal[0], cal[-1])
            row[label] = rr.summary
            if label == "IS":
                row["IS_rets"] = rr.daily_rets
                row["IS_curve"] = rr.curve
            if label == "HO":
                row["HO_rets"] = rr.daily_rets
                row["HO_curve"] = rr.curve
                row["HO_last_w"] = rr.last_weights
            if label == "EARLY":
                row["EARLY_rets"] = rr.daily_rets
        variant_rows.append(row)
        print(f"    IS sh={row['IS']['sharpe']:.2f} dd={row['IS']['max_drawdown']*100:.1f}% | HO sh={row['HO']['sharpe']:.2f} dd={row['HO']['max_drawdown']*100:.1f}% | EARLY sh={row['EARLY']['sharpe']:.2f}")

    # pick on IS: maximize sharpe + 0.3*(dd improvement vs baseline xs126_top3)
    baseline = next(r for r in variant_rows if r["name"] == "xs126_top3_ew_daily")
    base_dd = baseline["IS"]["max_drawdown"]
    scored = []
    for r in variant_rows:
        sh = r["IS"]["sharpe"]
        dd = r["IS"]["max_drawdown"]
        if sh != sh:
            continue
        score = sh + 0.3 * max(0.0, dd - base_dd)
        # prefer MaxDD better than -60% on IS if possible
        if dd > -0.60:
            score += 0.05
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    winner = scored[0][1]
    results["variant_trials"] = [{"name": r["name"], "IS": r["IS"], "HO": r["HO"], "EARLY": r["EARLY"]} for r in variant_rows]
    results["winner"] = {"name": winner["name"], "spec": winner["spec"],
                         "IS": winner["IS"], "HO": winner["HO"], "EARLY": winner["EARLY"],
                         "is_score": scored[0][0]}
    print("WINNER", winner["name"])

    n_trials = len(VARIANT_SPECS)
    haircut_sh = winner["HO"]["sharpe"] / math.sqrt(math.log(max(2, n_trials)))

    # ===== 4) Options translation for winner =====
    print("=== Options translation (ESTIMATED) ===", flush=True)
    # collect weekly picks on HO for winner
    picks_series = []
    hist = histories
    didx = date_index
    spec = winner["spec"]
    last_reb = -10**9
    # rebuild pick schedule on full 2018+ for options + shares compare on HO window
    port_dates = cal_is + cal_ho
    pending_w = {}
    for di, date in enumerate(port_dates):
        i_by = {s: didx[s][date] for s in hist if date in didx[s]}
        if (di - last_reb) >= spec["reb"]:
            last_reb = di
            w = xs_signal(hist, i_by, spec["lb"], spec["top"], inv_vol=spec["inv_vol"],
                          buckets=buckets_map, max_per_bucket=spec["mpb"])
            if spec["pvol"]:
                w = scale_to_pvol(w, hist, i_by, spec["pvol"])
            pending_w = w
            if date >= cal_ho[0]:
                picks_series.append({"date": date, "weights": dict(w)})

    opt_results = {}
    for iv_mult in (1.0, 1.2, 1.4):
        for long_mode in ("atm", "plus5", "path"):
            curve, trades = simulate_debit_spreads(
                picks_series, hist, didx, account0=ACCOUNT0, iv_mult=iv_mult,
                width=0.10, haircut=0.08, hold_days=10, long_mode=long_mode,
            )
            if not curve:
                continue
            rets=[]
            eqs=[e["equity"] for e in curve]
            for i in range(1,len(eqs)):
                if eqs[i-1]>0: rets.append(eqs[i]/eqs[i-1]-1)
            summ = summarize(curve, [Trade(t["entry_date"],t["exit_date"],t["symbol"],t["side"],t["qty"],t["entry_price"],t["exit_price"],t["pnl"]) for t in trades], ACCOUNT0, len(curve))
            key = f"iv{iv_mult}_{long_mode}"
            opt_results[key] = {"summary": summ, "n_trades": len(trades), "final": eqs[-1] if eqs else ACCOUNT0,
                                "label": "ESTIMATED BS debit spreads — NOT real option prices"}
            print(f"  {key}: final=${eqs[-1]:.2f} CAGR={summ['cagr']*100:.1f}% trades={len(trades)}")
    results["options_estimated"] = opt_results

    # Shares path on HO with $115, fractional OK, winner spec
    print("=== Shares $115 HO ===", flush=True)
    rr115 = run_variant(histories, date_index, cal_ho, winner["spec"], cost, ACCOUNT0, buckets_map, cal_ho[0], cal_ho[-1])
    results["shares_115_HO"] = rr115.summary
    print(f"  shares HO: final=${rr115.summary['end_equity']:.2f} CAGR={rr115.summary['cagr']*100:.1f}% DD={rr115.summary['max_drawdown']*100:.1f}%")

    # ===== 5) Monte Carlo =====
    print("=== Monte Carlo ===", flush=True)
    # use IS+HO combined rets from winner full 2018-2026
    rr_all = run_variant(histories, date_index, cal_is + cal_ho, winner["spec"], cost, 10_000.0, buckets_map, cal_is[0], cal_ho[-1])
    rets_all = rr_all.daily_rets
    mc_tables = {}
    for dep in (0, 100, 500, 1000):
        mc_tables[f"deposit_{dep}"] = millionaire_mc(
            rets_all, start=ACCOUNT0, deposits_monthly=float(dep), haircut_edge=0.5,
            n_paths=2500, years=(1, 3, 5, 10), seed=11 + dep,
        )
        print(f"  deposit ${dep}/mo haircut P(1M in 10y)={mc_tables[f'deposit_{dep}']['haircut']['10']['prob_1M']*100:.2f}% ruin80={mc_tables[f'deposit_{dep}']['haircut']['10']['prob_dd80']*100:.1f}%")
    results["montecarlo"] = mc_tables

    # ===== 6) Outputs =====
    series = {
        f"winner_{winner['name']}_ISHO": [{"date": e["date"], "equity": e["equity"]} for e in rr_all.curve],
        f"winner_{winner['name']}_HO_115": [{"date": e["date"], "equity": e["equity"]} for e in rr115.curve],
        "baseline_xs126_top3_ISHO": [{"date": e["date"], "equity": e["equity"]} for e in
            run_variant(histories, date_index, cal_is+cal_ho, VARIANT_SPECS[0], cost, 10_000.0, buckets_map, cal_is[0], cal_ho[-1]).curve],
    }
    spy_c, _ = buy_hold_equal({"SPY": spy}, {"SPY": spy_idx}, cal_is+cal_ho, ["SPY"], CostModel(5), 10_000.0, 1.0, cal_is[0], cal_ho[-1])
    series["SPY_buy_hold"] = [{"date": e["date"], "equity": e["equity"]} for e in spy_c]
    chart.write_equity_csv(run_dir / f"{base}_equity.csv", series)

    with (run_dir / f"{base}_trades.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["entry_date","exit_date","symbol","side","qty","entry_price","exit_price","pnl"])
        w.writeheader()
        for t in rr_all.trades:
            w.writerow({"entry_date": t.entry_date, "exit_date": t.exit_date, "symbol": t.symbol,
                        "side": t.side, "qty": t.qty, "entry_price": t.entry_price,
                        "exit_price": t.exit_price, "pnl": t.pnl})

    # montecarlo tidy: year, series, p10, p50, p90, prob_1M, etc.
    with (run_dir / f"{base}_montecarlo.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = ["years","deposit_monthly","case","prob_1k","prob_10k","prob_100k","prob_1M","prob_dd80","p10_final","p50_final","p90_final"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for dep_k, tab in mc_tables.items():
            dep = int(dep_k.split("_")[1])
            for case in ("raw", "haircut"):
                for y, m in tab[case].items():
                    w.writerow({
                        "years": y, "deposit_monthly": dep, "case": case,
                        "prob_1k": m["prob_1k"], "prob_10k": m["prob_10k"],
                        "prob_100k": m["prob_100k"], "prob_1M": m["prob_1M"],
                        "prob_dd80": m["prob_dd80"],
                        "p10_final": m["p10_final"], "p50_final": m["p50_final"], "p90_final": m["p90_final"],
                    })

    # Predict this week
    last_date = cal_ho[-1]
    i_by = {s: date_index[s][last_date] for s in histories if last_date in date_index[s]}
    w_now = xs_signal(histories, i_by, winner["spec"]["lb"], winner["spec"]["top"],
                      inv_vol=winner["spec"]["inv_vol"], buckets=buckets_map,
                      max_per_bucket=winner["spec"]["mpb"])
    if winner["spec"]["pvol"]:
        w_now = scale_to_pvol(w_now, histories, i_by, winner["spec"]["pvol"])
    predict = {"as_of": last_date, "strategy": winner["name"], "account": ACCOUNT0, "cash_approx": 2.40,
               "picks": [], "notes": ["FREEZE — do not place orders", "earnings skip NOT implemented (no free reliable calendar)"]}
    for sym, wt in sorted(w_now.items(), key=lambda x: -x[1]):
        px = histories[sym][i_by[sym]]["close"]
        dollars = min(0.50 * ACCOUNT0, wt * ACCOUNT0)
        shares = dollars / px
        # spread affordability
        rv = realized_vol([b["close"] for b in histories[sym][: i_by[sym] + 1]], 20) or 0.5
        iv = rv * 1.2
        debit = max(0.05, bs_call(px, round(px), 10/252, iv) - bs_call(px, round(px*1.10), 10/252, iv)) * 1.08
        cost1 = debit * 100
        n_aff = int(min(0.50 * ACCOUNT0, max(0, 2.40 + dollars)) // cost1) if cost1 > 0 else 0
        # with only $2.40 cash, realistically need to free capital — note
        predict["picks"].append({
            "symbol": sym, "weight": wt, "last_close": px,
            "shares_frac": shares, "dollars_at_50pct_cap": dollars,
            "est_spread_debit": debit, "est_cost_1_spread": cost1,
            "whole_spreads_affordable_if_capital_freed": n_aff,
            "affordable_with_2_40_cash_only": cost1 <= 2.40,
        })
    (run_dir / "predict_week.json").write_text(json.dumps(predict, indent=2))
    results["predict_week"] = predict
    results["earnings_note"] = "No free reliable earnings calendar wired; skip-earnings rule NOT applied — Tim should check manually."
    results["n_variant_trials"] = n_trials
    results["haircut_ho_sharpe"] = haircut_sh
    results["elapsed_s"] = round(time.time() - t0, 1)

    meta = {
        "base_name": base, "run_id": run_id, "run_at_et": run_at.isoformat(timespec="seconds"),
        "timezone": "America/New_York", "code_version": CODE_VERSION + "+momentum-tradeable",
        "simulated_vs_real": "simulated", "fees_included": True,
        "account_start_assumed": ACCOUNT0,
        "units": {"equity": "USD", "drawdown_pct": "negative fraction", "montecarlo probs": "0-1 fractions"},
        "files": [f"{base}_equity.csv", f"{base}_trades.csv", f"{base}_montecarlo.csv", f"{base}_meta.json", "REPORT.md", "predict_week.json"],
        "winner": winner["name"],
        "freeze": True,
    }
    chart.write_meta(run_dir / f"{base}_meta.json", meta)
    # strip heavy curves from results json
    slim = {k: v for k, v in results.items() if k not in ()}
    for row in slim.get("variant_trials", []):
        pass
    (run_dir / "results.json").write_text(json.dumps(slim, indent=2, default=str))

    # REPORT
    def pct(x):
        return "n/a" if x != x else f"{x*100:.2f}%"
    def sh(x):
        return "n/a" if x != x else f"{x:.2f}"
    lines = [
        f"# Momentum tradeable — `{run_id}`",
        "",
        f"**Tim's ask:** become a millionaire — honest version for a **${ACCOUNT0:.0f}** account.",
        f"Freeze on. No orders placed. Estimated options are **not** real fills.",
        "",
        "## 1) Fixed strategy-DD overlay",
        "Re-entry now uses **shadow (always-on) equity DD ≥ −6%** and/or SPY > SMA50, not frozen flat equity.",
        "",
    ]
    for mode, windows in dd_fix.items():
        lines.append(f"### `{mode}`")
        lines.append("| Window | CAGR | Sharpe | MaxDD | % cash | Switches |")
        lines.append("|---|---|---|---|---|---|")
        for wname, m in windows.items():
            lines.append(f"| {wname} | {pct(m['cagr'])} | {sh(m['sharpe'])} | {pct(m['max_drawdown'])} | {pct(m['pct_time_cash'])} | {m['n_switches']} |")
        lines.append("")

    lines += ["## 2) Frozen rules — early OOS (2010–2017) vs later",
              "Early test uses names that existed by mid-2010 (survivorship still present but better than post-only).",
              "",
              "| Rule | EARLY CAGR | EARLY Sh | EARLY DD | IS Sh | HO Sh |",
              "|---|---|---|---|---|---|"]
    for name, periods in frozen_res.items():
        e, i, h = periods.get("EARLY_2010_2017", {}), periods.get("IS_2018_2025", {}), periods.get("HO_2025_2026", {})
        lines.append(f"| {name} | {pct(e.get('cagr', float('nan')))} | {sh(e.get('sharpe', float('nan')))} | {pct(e.get('max_drawdown', float('nan')))} | {sh(i.get('sharpe', float('nan')))} | {sh(h.get('sharpe', float('nan')))} |")
    lines += ["", "### Rolling-origin WF (xs126 top3, calendar years)",
              "| Year | CAGR | Sharpe | MaxDD | n_syms |", "|---|---|---|---|---|"]
    for row in rolling:
        lines.append(f"| {row['test_year']} | {pct(row['cagr'])} | {sh(row['sharpe'])} | {pct(row['max_drawdown'])} | {row['n_syms']} |")

    lines += ["", f"## 3) Winner after DD-control hunt (`{winner['name']}`)",
              f"- Trials logged: **{n_trials}**",
              f"- Spec: `{json.dumps(winner['spec'])}`",
              "",
              "| Window | CAGR | Sharpe | Sortino | MaxDD | Trades |",
              "|---|---|---|---|---|---|",
              f"| IS | {pct(winner['IS']['cagr'])} | {sh(winner['IS']['sharpe'])} | {sh(winner['IS']['sortino'])} | {pct(winner['IS']['max_drawdown'])} | {winner['IS']['n_trades']} |",
              f"| HO 2025-26 | {pct(winner['HO']['cagr'])} | {sh(winner['HO']['sharpe'])} | {sh(winner['HO']['sortino'])} | {pct(winner['HO']['max_drawdown'])} | {winner['HO']['n_trades']} |",
              f"| EARLY 2010-17 | {pct(winner['EARLY']['cagr'])} | {sh(winner['EARLY']['sharpe'])} | {sh(winner['EARLY']['sortino'])} | {pct(winner['EARLY']['max_drawdown'])} | {winner['EARLY']['n_trades']} |",
              f"",
              f"- Haircut HO Sharpe / sqrt(log({n_trials})): **{haircut_sh:.2f}**",
              f"- $115 HO shares path: final **${rr115.summary['end_equity']:.2f}**, CAGR {pct(rr115.summary['cagr'])}, MaxDD {pct(rr115.summary['max_drawdown'])}",
              "",
              "## Plain-English weekly rules (winner)",
              "1. Once a week (or every 5 trading days), rank the high-vol universe by ~6-month return (skip most recent day).",
              "2. Buy the top names per the winner spec (see predict_week.json), ≤50% per ticker.",
              "3. Equal-weight or inv-vol / vol-target as specified; diversify buckets if mpb set.",
              "4. Check earnings manually before entry (auto-skip not available on free data).",
              "5. Under freeze: bot proposes only; Tim clicks.",
              "",
              "## 4) Options translation (ESTIMATED)",
              "BS debit spreads, IV=RV20×mult, 10% width, 8% haircut/side, 10 trading-day hold, whole contracts, ≤50%/ticker.",
              "With ~$2.40 cash, most spreads are **not affordable** until capital is freed or deposits arrive.",
              ""]
    for k, v in opt_results.items():
        lines.append(f"- `{k}`: final ${v['final']:.2f}, CAGR {pct(v['summary']['cagr'])}, trades {v['n_trades']}")

    lines += ["", "## 5) Millionaire reality check (block-bootstrap)",
              "Main case = **haircut** (daily returns × 0.5). Optimistic = raw backtest returns.",
              "Start $115.", ""]
    for dep in (0, 100, 500, 1000):
        tab = mc_tables[f"deposit_{dep}"]["haircut"]
        lines.append(f"### Haircut case — ${dep}/mo deposits")
        lines.append("| Years | P($1k) | P($10k) | P($100k) | P($1M) | P(−80% DD) | p50 final |")
        lines.append("|---|---|---|---|---|---|---|")
        for y in ("1", "3", "5", "10"):
            m = tab[y]
            lines.append(f"| {y} | {m['prob_1k']*100:.1f}% | {m['prob_10k']*100:.1f}% | {m['prob_100k']*100:.1f}% | {m['prob_1M']*100:.2f}% | {m['prob_dd80']*100:.1f}% | ${m['p50_final']:.0f} |")
        lines.append("")

    lines += ["## This week's picks", json.dumps(predict, indent=2), "",
              f"## Files", f"`{run_dir}/`",
              f"- `{base}_equity.csv`", f"- `{base}_trades.csv`", f"- `{base}_montecarlo.csv`",
              f"- `{base}_meta.json`", ""]
    (run_dir / "REPORT.md").write_text("\n".join(lines))

    # Update PredictDayHighVol-style helper
    pred_path = ROOT / "Backtest" / "PredictWeekTradeable.py"
    pred_path.write_text('''"""Print this week's tradeable momentum picks for ~$115 account (read-only)."""
from __future__ import annotations
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
# default to latest mt_ run
runs = sorted((ROOT/"out").glob("mt_*"))
if not runs:
    raise SystemExit("no mt_ runs")
pred = json.loads((runs[-1]/"predict_week.json").read_text())
print(json.dumps(pred, indent=2))
''')
    print("\nDONE", run_dir, "elapsed", round(time.time()-t0,1))


if __name__ == "__main__":
    main()
