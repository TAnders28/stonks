"""Option structures (SYNTHETIC strikes/expiries; never historical observations).

Strikes are placed in units of the model expected move EM = S * atm_iv * sqrt(T) and rounded to a synthetic
listing grid. Legs are per-share units: option qty +1 = one long contract (x100), stock qty +1 = 100 shares.
"""
from __future__ import annotations
import math

# name: (bias needed, description)
STRUCTURES = {
    "long_call_atm":     ("bull", "buy ATM call"),
    "long_call_otm":     ("bull", "buy call at S+0.5EM"),
    "long_put_atm":      ("bear", "buy ATM put"),
    "bull_call_debit":   ("bull", "buy ATM call / sell S+0.5EM call"),
    "bear_put_debit":    ("bear", "buy ATM put / sell S-0.5EM put"),
    "bull_put_credit":   ("bull", "sell S-0.5EM put / buy S-1.0EM put"),
    "bear_call_credit":  ("bear", "sell S+0.5EM call / buy S+1.0EM call"),
    "iron_condor":       ("any",  "sell +-0.75EM, buy +-1.25EM (put+call credit spreads)"),
    "call_butterfly":    ("any",  "long 1 K-0.5EM, short 2 ATM, long 1 K+0.5EM calls"),
    "long_straddle":     ("any",  "buy ATM call + ATM put"),
    "long_strangle":     ("any",  "buy S+0.5EM call + S-0.5EM put"),
    "covered_call":      ("bull", "buy 100 shares, sell S+0.5EM call (collateralized comparator)"),
    "cash_secured_put":  ("bull", "sell S-0.5EM put, hold K*100 cash (collateralized comparator)"),
}
REJECTED = {
    "calendar_spread": "needs a term-structure model for the far leg; trailing-RV cannot validate it -> not accurately supported in phase 1",
    "diagonal_spread": "same as calendar (two maturities, unobservable term structure/skew dynamics)",
    "naked_short_call": "uncovered unlimited-risk selling excluded from small-account candidates",
    "short_straddle/strangle": "uncovered large/unlimited risk; excluded",
    "ratio_spreads": "uncovered tail risk; excluded from phase-1 grid",
}


def StrikeStep(S: float) -> float:
    if S < 25:
        return 0.5
    if S < 200:
        return 1.0
    return 5.0


def RoundStrike(x: float, S: float) -> float:
    st = StrikeStep(S)
    return max(st, round(x / st) * st)


def BuildLegs(name: str, S: float, atm_iv: float, T: float) -> list[dict]:
    em = S * atm_iv * math.sqrt(max(T, 1e-6))
    st = StrikeStep(S)
    R = lambda x: RoundStrike(x, S)
    atm = R(S)
    def off(k):  # strike k*EM away, at least one step from ATM
        v = R(S + k * em)
        if k > 0 and v <= atm: v = atm + st * max(1, round(abs(k)))
        if k < 0 and v >= atm: v = atm - st * max(1, round(abs(k)))
        return max(st, v)
    C = lambda K, q: {"kind": "C", "K": K, "qty": q}
    P = lambda K, q: {"kind": "P", "K": K, "qty": q}
    if name == "long_call_atm": return [C(atm, 1)]
    if name == "long_call_otm": return [C(off(0.5), 1)]
    if name == "long_put_atm": return [P(atm, 1)]
    if name == "bull_call_debit": return [C(atm, 1), C(off(0.5), -1)]
    if name == "bear_put_debit": return [P(atm, 1), P(off(-0.5), -1)]
    if name == "bull_put_credit":
        a = off(-0.5); b = min(off(-1.0), a - st); return [P(a, -1), P(max(st, b), 1)]
    if name == "bear_call_credit":
        a = off(0.5); b = max(off(1.0), a + st); return [C(a, -1), C(b, 1)]
    if name == "iron_condor":
        ps = off(-0.75); pl = min(off(-1.25), ps - st); cs = off(0.75); cl = max(off(1.25), cs + st)
        return [P(ps, -1), P(max(st, pl), 1), C(cs, -1), C(cl, 1)]
    if name == "call_butterfly":
        lo = off(-0.5); hi = atm + (atm - lo)
        return [C(lo, 1), C(atm, -2), C(hi, 1)]
    if name == "long_straddle": return [C(atm, 1), P(atm, 1)]
    if name == "long_strangle": return [C(off(0.5), 1), P(off(-0.5), 1)]
    if name == "covered_call": return [{"kind": "S", "K": 0.0, "qty": 1}, C(off(0.5), -1)]
    if name == "cash_secured_put": return [P(off(-0.5), -1)]
    raise KeyError(name)
