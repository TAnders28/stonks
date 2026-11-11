"""High-vol strategy signal families. All use bars[0..i] only (no lookahead)."""
from __future__ import annotations
import math
from typing import Callable


def sma(xs: list[float], n: int) -> float | None:
    if len(xs) < n:
        return None
    return sum(xs[-n:]) / n


def rsi(closes: list[float], n: int = 2) -> float | None:
    if len(closes) < n + 1:
        return None
    gains = losses = 0.0
    for i in range(-n, 0):
        d = closes[i] - closes[i - 1]
        if d >= 0:
            gains += d
        else:
            losses -= d
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))


def realized_vol(closes: list[float], n: int = 20) -> float | None:
    if len(closes) < n + 1:
        return None
    rets = []
    for i in range(-n, 0):
        if closes[i - 1] > 0:
            rets.append(math.log(closes[i] / closes[i - 1]))
    if len(rets) < n:
        return None
    m = sum(rets) / len(rets)
    var = sum((r - m) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(252)


def ibs(bar: dict) -> float:
    # Internal Bar Strength
    rng = bar["high"] - bar["low"]
    if rng <= 0:
        return 0.5
    return (bar["close"] - bar["low"]) / rng


def atr(bars: list[dict], n: int = 14) -> float | None:
    if len(bars) < n + 1:
        return None
    trs = []
    for i in range(-n, 0):
        h, l, pc = bars[i]["high"], bars[i]["low"], bars[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return sum(trs) / n


# ---- factories return signal_fn(panel, i) -> dict[sym, weight] ----
# panel: {sym: list[dict bars aligned?]} — we use per-symbol histories with date index map

def ts_momentum_volscale(fast: int, slow: int, vol_lookback: int, target_vol: float, max_w: float):
    """Time-series: long if SMA_fast > SMA_slow; weight = min(max_w, target_vol/realized_vol)."""
    name = f"ts_mom_f{fast}_s{slow}_tv{target_vol:.0%}_vw{vol_lookback}"

    def signal(histories: dict[str, list[dict]], i_by_sym: dict[str, int], regime_ok: bool) -> dict[str, float]:
        if not regime_ok:
            return {}
        w = {}
        for sym, bars in histories.items():
            i = i_by_sym.get(sym)
            if i is None or i < slow:
                continue
            closes = [b["close"] for b in bars[: i + 1]]
            sf, ss = sma(closes, fast), sma(closes, slow)
            if sf is None or ss is None or sf <= ss:
                continue
            vol = realized_vol(closes, vol_lookback) or target_vol
            wt = min(max_w, target_vol / max(vol, 1e-6))
            w[sym] = wt
        # equalize if many names: scale so sum <= 1 while respecting max_w
        s = sum(w.values())
        if s > 1.0:
            w = {k: v / s for k, v in w.items()}
            w = {k: min(max_w, v) for k, v in w.items()}
            s = sum(w.values())
            if s > 1.0:
                w = {k: v / s for k, v in w.items()}
        return w

    signal.name = name
    return signal


def cross_sectional_momentum(lookback: int, top_n: int, max_w: float, skip_days: int = 1):
    """Cross-sectional: rank by lookback return (skip most recent skip_days), equal-weight top_n."""
    name = f"xs_mom_lb{lookback}_top{top_n}_skip{skip_days}"

    def signal(histories, i_by_sym, regime_ok):
        if not regime_ok:
            return {}
        scores = []
        for sym, bars in histories.items():
            i = i_by_sym.get(sym)
            if i is None or i < lookback + skip_days:
                continue
            c_now = bars[i - skip_days]["close"]
            c_past = bars[i - skip_days - lookback]["close"]
            if c_past <= 0:
                continue
            scores.append((sym, c_now / c_past - 1))
        scores.sort(key=lambda x: x[1], reverse=True)
        picks = [s for s, _ in scores[:top_n] if _ > 0]
        if not picks:
            return {}
        wt = min(max_w, 1.0 / len(picks))
        return {s: wt for s in picks}

    signal.name = name
    return signal


def rsi2_mr_in_uptrend(rsi_n: int, rsi_buy: float, trend_sma: int, max_w: float, top_n: int = 5):
    """Mean reversion: RSI(n) < rsi_buy AND price > SMA(trend); equal weight among triggers, cap top_n."""
    name = f"rsi{rsi_n}_mr_lt{rsi_buy}_trend{trend_sma}_top{top_n}"

    def signal(histories, i_by_sym, regime_ok):
        if not regime_ok:
            return {}
        picks = []
        for sym, bars in histories.items():
            i = i_by_sym.get(sym)
            if i is None or i < max(trend_sma, rsi_n + 2):
                continue
            closes = [b["close"] for b in bars[: i + 1]]
            r = rsi(closes, rsi_n)
            sm = sma(closes, trend_sma)
            if r is None or sm is None:
                continue
            if r < rsi_buy and closes[-1] > sm:
                picks.append((sym, r))
        picks.sort(key=lambda x: x[1])  # most oversold first
        picks = [s for s, _ in picks[:top_n]]
        if not picks:
            return {}
        wt = min(max_w, 1.0 / len(picks))
        return {s: wt for s in picks}

    signal.name = name
    return signal


def ibs_bounce(trend_sma: int, ibs_max: float, max_w: float, top_n: int = 5):
    name = f"ibs_lt{ibs_max}_trend{trend_sma}_top{top_n}"

    def signal(histories, i_by_sym, regime_ok):
        if not regime_ok:
            return {}
        picks = []
        for sym, bars in histories.items():
            i = i_by_sym.get(sym)
            if i is None or i < trend_sma:
                continue
            closes = [b["close"] for b in bars[: i + 1]]
            sm = sma(closes, trend_sma)
            if sm is None or closes[-1] <= sm:
                continue
            score = ibs(bars[i])
            if score <= ibs_max:
                picks.append((sym, score))
        picks.sort(key=lambda x: x[1])
        picks = [s for s, _ in picks[:top_n]]
        if not picks:
            return {}
        wt = min(max_w, 1.0 / len(picks))
        return {s: wt for s in picks}

    signal.name = name
    return signal


def donchian_atr(channel: int, atr_n: int, atr_stop_mult: float, max_w: float, top_n: int = 5):
    """Breakout: close > max(high[-channel:-1]); hold until close < entry - atr_stop*ATR (approx via weight 0).
    Simplified: weight if close is N-day high; else 0. Stop handled by requiring close > SMA or channel mid.
    """
    name = f"donchian{channel}_atr{atr_n}x{atr_stop_mult}_top{top_n}"

    def signal(histories, i_by_sym, regime_ok):
        if not regime_ok:
            return {}
        picks = []
        for sym, bars in histories.items():
            i = i_by_sym.get(sym)
            if i is None or i < channel + atr_n:
                continue
            window = bars[i - channel : i]  # exclude today for breakout level (classic)
            level = max(b["high"] for b in window)
            if bars[i]["close"] > level:
                a = atr(bars[: i + 1], atr_n)
                if a:
                    picks.append((sym, (bars[i]["close"] - level) / a))
        picks.sort(key=lambda x: x[1], reverse=True)
        picks = [s for s, _ in picks[:top_n]]
        if not picks:
            return {}
        wt = min(max_w, 1.0 / len(picks))
        return {s: wt for s in picks}

    signal.name = name
    return signal


def bucket_rs_rotation(lookback: int, buckets: dict[str, list[str]], max_w: float, names_per_bucket: int = 2):
    """Pick best bucket by equal-weight lookback return, then top names inside it by RS."""
    name = f"bucket_rs_lb{lookback}_npb{names_per_bucket}"

    def signal(histories, i_by_sym, regime_ok):
        if not regime_ok:
            return {}
        bucket_scores = []
        for bname, syms in buckets.items():
            rets = []
            for sym in syms:
                bars = histories.get(sym)
                i = i_by_sym.get(sym)
                if bars is None or i is None or i < lookback:
                    continue
                c0, c1 = bars[i - lookback]["close"], bars[i]["close"]
                if c0 > 0:
                    rets.append(c1 / c0 - 1)
            if len(rets) >= 2:
                bucket_scores.append((bname, sum(rets) / len(rets)))
        if not bucket_scores:
            return {}
        bucket_scores.sort(key=lambda x: x[1], reverse=True)
        best = bucket_scores[0][0]
        if bucket_scores[0][1] <= 0:
            return {}
        # top names in best bucket
        scores = []
        for sym in buckets[best]:
            bars = histories.get(sym)
            i = i_by_sym.get(sym)
            if bars is None or i is None or i < lookback:
                continue
            c0, c1 = bars[i - lookback]["close"], bars[i]["close"]
            if c0 > 0:
                scores.append((sym, c1 / c0 - 1))
        scores.sort(key=lambda x: x[1], reverse=True)
        picks = [s for s, r in scores[:names_per_bucket] if r > 0]
        if not picks:
            return {}
        wt = min(max_w, 1.0 / len(picks))
        return {s: wt for s in picks}

    signal.name = name
    return signal


def build_variant_grid(trade_buckets: dict[str, list[str]], max_w: float = 0.50) -> list:
    """Pre-register a finite set of variants (logged for multiple-testing)."""
    variants = []
    # TS momentum vol-scaled
    for fast, slow in [(10, 50), (20, 50), (20, 100), (50, 200)]:
        for tv in [0.20, 0.40]:
            variants.append(("ts_mom", ts_momentum_volscale(fast, slow, 20, tv, max_w)))
    # XS momentum
    for lb in [21, 63, 126]:
        for top in [3, 5]:
            variants.append(("xs_mom", cross_sectional_momentum(lb, top, max_w, skip_days=1)))
    # RSI2 MR
    for rb in [5, 10]:
        for trend in [50, 100]:
            for top in [3, 5]:
                variants.append(("rsi_mr", rsi2_mr_in_uptrend(2, rb, trend, max_w, top)))
    # IBS
    for ib in [0.2, 0.3]:
        for trend in [50, 100]:
            variants.append(("ibs", ibs_bounce(trend, ib, max_w, top_n=5)))
    # Donchian
    for ch in [20, 55]:
        variants.append(("donchian", donchian_atr(ch, 14, 2.0, max_w, top_n=5)))
    # Bucket RS
    for lb in [21, 63]:
        variants.append(("bucket_rs", bucket_rs_rotation(lb, trade_buckets, max_w, names_per_bucket=3)))
    return variants
