"""Option valuation: Black-Scholes-Merton (European, continuous dividend yield q) + Greeks,
CRR binomial (European/American, continuous q and/or discrete cash dividends via escrowed model),
expiry payoffs and no-arbitrage bounds.

Units: price per share (x100 multiplier applied by callers). Greeks: delta per $1, gamma per $1 per $1,
vega per 1 vol POINT (0.01), theta per CALENDAR day, rho per 1 percentage point of r.
"""
from __future__ import annotations
import math
import numpy as np

SQRT2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def NormCdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def NormPdf(x: float) -> float:
    return INV_SQRT_2PI * math.exp(-0.5 * x * x)


def Intrinsic(S: float, K: float, kind: str) -> float:
    return max(S - K, 0.0) if kind == "C" else max(K - S, 0.0)


def BsPrice(S, K, T, r, q, sigma, kind):
    if T <= 0:
        return Intrinsic(S, K, kind)
    df_r, df_q = math.exp(-r * T), math.exp(-q * T)
    if sigma <= 0:
        fwd = S * df_q - K * df_r
        return max(fwd, 0.0) if kind == "C" else max(-fwd, 0.0)
    sv = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sv
    d2 = d1 - sv
    if kind == "C":
        return S * df_q * NormCdf(d1) - K * df_r * NormCdf(d2)
    return K * df_r * NormCdf(-d2) - S * df_q * NormCdf(-d1)


def BsGreeks(S, K, T, r, q, sigma, kind) -> dict:
    if T <= 0 or sigma <= 0:
        itm = (S > K) if kind == "C" else (S < K)
        d = (1.0 if itm else 0.0) * (1 if kind == "C" else -1)
        return {"price": BsPrice(S, K, T, r, q, sigma, kind), "delta": d, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    df_r, df_q = math.exp(-r * T), math.exp(-q * T)
    sq = math.sqrt(T); sv = sigma * sq
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sv
    d2 = d1 - sv
    pdf = NormPdf(d1)
    gamma = df_q * pdf / (S * sv)
    vega = S * df_q * pdf * sq / 100.0
    if kind == "C":
        delta = df_q * NormCdf(d1)
        theta = (-S * df_q * pdf * sigma / (2 * sq) - r * K * df_r * NormCdf(d2) + q * S * df_q * NormCdf(d1)) / 365.0
        rho = K * T * df_r * NormCdf(d2) / 100.0
    else:
        delta = -df_q * NormCdf(-d1)
        theta = (-S * df_q * pdf * sigma / (2 * sq) + r * K * df_r * NormCdf(-d2) - q * S * df_q * NormCdf(-d1)) / 365.0
        rho = -K * T * df_r * NormCdf(-d2) / 100.0
    return {"price": BsPrice(S, K, T, r, q, sigma, kind), "delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def BinomialPrice(S, K, T, r, q, sigma, kind, american=True, steps=500, divs=None):
    """CRR tree. divs: list of (t_years, cash_amount) discrete dividends (escrowed-dividend model:
    tree on S minus PV(divs before T); exercise value adds back PV of not-yet-paid dividends)."""
    if T <= 0:
        return Intrinsic(S, K, kind)
    divs = [(t, d) for t, d in (divs or []) if 0 < t <= T]
    pv_all = sum(d * math.exp(-r * t) for t, d in divs)
    s_star = S - pv_all
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt)); dn = 1.0 / u
    a = math.exp((r - q) * dt)
    p = (a - dn) / (u - dn)
    disc = math.exp(-r * dt)
    j = np.arange(steps + 1)
    st = s_star * u ** (steps - 2 * j)
    vals = np.maximum(st - K, 0.0) if kind == "C" else np.maximum(K - st, 0.0)
    for i in range(steps - 1, -1, -1):
        vals = disc * (p * vals[:-1] + (1 - p) * vals[1:])
        if american:
            t = i * dt
            pv_rem = sum(d * math.exp(-r * (td - t)) for td, d in divs if td > t)
            st = s_star * u ** (i - 2 * np.arange(i + 1)) + pv_rem
            ex = np.maximum(st - K, 0.0) if kind == "C" else np.maximum(K - st, 0.0)
            vals = np.maximum(vals, ex)
    return float(vals[0])


def PriceBounds(S, K, T, r, q, kind):
    """European no-arbitrage bounds (lower, upper)."""
    df_r, df_q = math.exp(-r * T), math.exp(-q * T)
    if kind == "C":
        return max(S * df_q - K * df_r, 0.0), S * df_q
    return max(K * df_r - S * df_q, 0.0), K * df_r


def PackagePayoff(legs, S_T):
    """legs: iterable of dict(kind in C/P/S, K, qty) per share-equivalent unit (qty in packages; options x1 share).
    Stock leg value at expiry = S_T. Returns value per 1 share-unit of the package."""
    v = 0.0
    for lg in legs:
        if lg["kind"] == "S":
            v += lg["qty"] * S_T
        else:
            v += lg["qty"] * Intrinsic(S_T, lg["K"], lg["kind"])
    return v


def PayoffExtremes(legs):
    """(min, max) of expiry payoff over S_T in [0, inf); max may be inf (unbounded upside), min may be -inf."""
    ks = sorted({lg["K"] for lg in legs if lg["kind"] != "S"})
    hi = (max(ks) if ks else 1.0) * 10 + 10
    pts = [0.0] + ks + [hi]
    vals = [PackagePayoff(legs, x) for x in pts]
    slope_inf = PackagePayoff(legs, hi * 2) - PackagePayoff(legs, hi)
    mn, mx = min(vals), max(vals)
    if slope_inf < -1e-9:
        mn = -math.inf
    if slope_inf > 1e-9:
        mx = math.inf
    return mn, mx
