"""Trailing-only volatility estimates and LABELED implied-vol scenarios.

Historical implied vol is NOT observed here. We assume IV = (trailing realized vol) x IV-risk-premium multiplier,
with optional skew and an earnings-jump variance add-on. All scenario parameters are assumptions.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np

EWMA_LAMBDA = 0.94
VOL_FLOOR, VOL_CAP = 0.08, 3.0


def TrailingVol(closes: np.ndarray) -> dict:
    """Annualized vol arrays aligned to closes; value at t uses log returns up to and including t (known at close t)."""
    c = np.asarray(closes, float)
    lr = np.full(len(c), np.nan)
    lr[1:] = np.log(c[1:] / c[:-1])
    ew = np.full(len(c), np.nan); var = None
    for t in range(1, len(c)):
        x = lr[t]
        if not np.isfinite(x):
            ew[t] = ew[t - 1]; continue
        var = x * x if var is None else EWMA_LAMBDA * var + (1 - EWMA_LAMBDA) * x * x
        ew[t] = math.sqrt(var * 252)
    s = lambda w: np.array([np.nanstd(lr[max(1, t - w + 1):t + 1], ddof=1) * math.sqrt(252) if t >= w else np.nan for t in range(len(c))])
    rv20, rv60 = s(20), s(60)
    blend = np.where(np.isfinite(rv60), 0.5 * ew + 0.5 * rv60, ew)
    blend = np.clip(blend, VOL_FLOOR, VOL_CAP)
    return {"ewma": ew, "rv20": rv20, "rv60": rv60, "base": blend, "lr": lr}


@dataclass(frozen=True)
class IvScenario:
    name: str
    ivrp: float = 1.2          # IV / trailing RV multiplier (assumed)
    skew: float = 0.0          # IV change per 1 std of log-moneyness (positive => OTM puts richer, OTM calls cheaper)
    crush: float = 0.0         # extra entry-only IV multiplier on single-name/event tickers (IV crush stress)
    earn_mode: str = "skip"    # 'skip' (Tim's rule: no positions spanning known earnings) or 'price_jump'


def ImpliedVol(base_sigma, S, K, T, scen: IvScenario, event_var=0.0, at_entry=False, event_name=False):
    mult = scen.ivrp + (scen.crush if (at_entry and event_name) else 0.0)
    atm = max(VOL_FLOOR, base_sigma * mult)
    if T <= 0:
        return atm
    m = math.log(K / S) / (atm * math.sqrt(T))
    iv = atm * (1.0 - scen.skew * m)
    iv = min(max(iv, 0.5 * atm), 2.5 * atm)
    if event_var > 0:
        iv = math.sqrt(iv * iv + event_var / T)
    return iv
