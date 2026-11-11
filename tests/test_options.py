"""Unit tests: pricing, Greeks (finite differences), bounds, American cross-check, multi-leg payoffs,
cost model, collateral, day-trade guard. Run: .venv\\Scripts\\python -m unittest tests.test_options -v"""
import math, unittest
from Options.Pricing import BsPrice, BsGreeks, BinomialPrice, PriceBounds, PackagePayoff, PayoffExtremes, Intrinsic
from Options.Structures import BuildLegs, STRUCTURES
from Options.DayTradeGuard import DayTradeGuard
from Options.VolModel import IvScenario, ImpliedVol, TrailingVol
import numpy as np

CASES = [(100, 100, 0.25, 0.03, 0.0, 0.3), (100, 90, 0.04, 0.02, 0.01, 0.6), (50, 60, 1.0, 0.05, 0.02, 0.2), (680, 700, 10 / 365, 0.02, 0.0, 0.15)]


class TestPricing(unittest.TestCase):
    def test_put_call_parity(self):
        for S, K, T, r, q, v in CASES:
            c, p = BsPrice(S, K, T, r, q, v, "C"), BsPrice(S, K, T, r, q, v, "P")
            self.assertAlmostEqual(c - p, S * math.exp(-q * T) - K * math.exp(-r * T), places=8)

    def test_bounds(self):
        for S, K, T, r, q, v in CASES:
            for k in "CP":
                lo, hi = PriceBounds(S, K, T, r, q, k); px = BsPrice(S, K, T, r, q, v, k)
                self.assertGreaterEqual(px, lo - 1e-9); self.assertLessEqual(px, hi + 1e-9)

    def test_expiry_and_zero_vol(self):
        self.assertEqual(BsPrice(105, 100, 0, 0.02, 0, 0.3, "C"), 5)
        self.assertEqual(BsPrice(95, 100, 0, 0.02, 0, 0.3, "P"), 5)
        self.assertAlmostEqual(BsPrice(105, 100, 1, 0.0, 0, 0.0, "C"), 5)

    def test_greeks_vs_finite_differences(self):
        for S, K, T, r, q, v in CASES:
            for k in "CP":
                g = BsGreeks(S, K, T, r, q, v, k); h = S * 1e-4
                f = lambda s=S, t=T, rr=r, vv=v: BsPrice(s, K, t, rr, q, vv, k)
                self.assertAlmostEqual(g["delta"], (f(s=S + h) - f(s=S - h)) / (2 * h), places=5)
                self.assertAlmostEqual(g["gamma"], (f(s=S + h) - 2 * f() + f(s=S - h)) / h ** 2, delta=1e-4 * max(1, g["gamma"]))
                self.assertAlmostEqual(g["vega"], (f(vv=v + 1e-4) - f(vv=v - 1e-4)) / 2e-4 / 100, places=6)
                self.assertAlmostEqual(g["rho"], (f(rr=r + 1e-5) - f(rr=r - 1e-5)) / 2e-5 / 100, places=6)
                dt = 1 / 365 / 24
                self.assertAlmostEqual(g["theta"], -(f(t=T + dt) - f(t=T - dt)) / (2 * dt) / 365, places=5)

    def test_binomial_converges_to_bs_european(self):
        for S, K, T, r, q, v in CASES:
            for k in "CP":
                self.assertAlmostEqual(BinomialPrice(S, K, T, r, q, v, k, american=False, steps=800), BsPrice(S, K, T, r, q, v, k), delta=0.01 * max(1, BsPrice(S, K, T, r, q, v, k)))

    def test_american_ge_european_and_call_no_div_equal(self):
        for S, K, T, r, q, v in CASES:
            for k in "CP":
                am = BinomialPrice(S, K, T, r, q, v, k, american=True, steps=600)
                eu = BinomialPrice(S, K, T, r, q, v, k, american=False, steps=600)
                self.assertGreaterEqual(am + 1e-12, eu)
                self.assertGreaterEqual(am + 1e-9, Intrinsic(S, K, k))
            am_c = BinomialPrice(S, K, T, r, 0.0, v, "C", american=True, steps=600)
            self.assertAlmostEqual(am_c, BinomialPrice(S, K, T, r, 0.0, v, "C", american=False, steps=600), places=8)

    def test_two_week_american_put_premium_small(self):
        # documents the European approximation error for ~2-week options (r=2%)
        S, K, T, r, v = 100, 105, 10 / 365, 0.02, 0.4
        diff = BinomialPrice(S, K, T, r, 0, v, "P", True, 800) - BinomialPrice(S, K, T, r, 0, v, "P", False, 800)
        self.assertLess(diff, 0.02)

    def test_discrete_dividend_call_early_exercise(self):
        S, K, T, r, v = 100, 90, 0.25, 0.02, 0.25
        am = BinomialPrice(S, K, T, r, 0, v, "C", True, 600, divs=[(0.1, 3.0)])
        eu = BinomialPrice(S, K, T, r, 0, v, "C", False, 600, divs=[(0.1, 3.0)])
        self.assertGreater(am, eu)


class TestMultiLeg(unittest.TestCase):
    def test_vertical_payoff_bounded_by_width(self):
        legs = [{"kind": "C", "K": 100, "qty": 1}, {"kind": "C", "K": 105, "qty": -1}]
        mn, mx = PayoffExtremes(legs); self.assertEqual((mn, mx), (0.0, 5.0))
        for S in (50, 100, 102, 105, 500): self.assertTrue(0 <= PackagePayoff(legs, S) <= 5)

    def test_condor_and_butterfly(self):
        ic = [{"kind": "P", "K": 90, "qty": -1}, {"kind": "P", "K": 85, "qty": 1}, {"kind": "C", "K": 110, "qty": -1}, {"kind": "C", "K": 115, "qty": 1}]
        self.assertEqual(PayoffExtremes(ic), (-5.0, 0.0))
        bf = [{"kind": "C", "K": 95, "qty": 1}, {"kind": "C", "K": 100, "qty": -2}, {"kind": "C", "K": 105, "qty": 1}]
        self.assertEqual(PayoffExtremes(bf), (0.0, 5.0))

    def test_unlimited_risk_detected_and_covered_call_bounded(self):
        self.assertEqual(PayoffExtremes([{"kind": "C", "K": 100, "qty": -1}])[0], -math.inf)
        cc = [{"kind": "S", "K": 0, "qty": 1}, {"kind": "C", "K": 105, "qty": -1}]
        self.assertEqual(PayoffExtremes(cc), (0.0, 105.0))
        self.assertEqual(PayoffExtremes([{"kind": "P", "K": 95, "qty": -1}]), (-95.0, 0.0))

    def test_spread_value_within_bounds_all_structures(self):
        S, T, r = 100.0, 10 / 365, 0.02
        for name in STRUCTURES:
            legs = BuildLegs(name, S, 0.4, T)
            mn, mx = PayoffExtremes(legs)
            self.assertTrue(math.isfinite(mn), name)
            v = sum(lg["qty"] * (S if lg["kind"] == "S" else BsPrice(S, lg["K"], T, r, 0, 0.4, lg["kind"])) for lg in legs)
            self.assertGreaterEqual(v, mn * math.exp(-r * T) - 1e-9, name)
            self.assertLessEqual(v, mx + 1e-9, name)

    def test_cost_model_and_collateral(self):
        from Options import SynthBacktest as SB
        legs = BuildLegs("bull_put_credit", 100.0, 0.4, 10 / 365)
        mids = [BsPrice(100, lg["K"], 10 / 365, 0.02, 0, 0.4, lg["kind"]) for lg in legs]
        open_flow = SB.PackageFill("SPY", legs, mids, 100, 1.0, True)
        close_flow = SB.PackageFill("SPY", legs, mids, 100, 1.0, False)
        mid_flow = sum(lg["qty"] * m for lg, m in zip(legs, mids)) * 100
        self.assertGreater(open_flow, mid_flow)          # credit received is less than mid
        self.assertGreater(close_flow, -mid_flow)        # costs to close exceed mid
        self.assertGreater(SB.PackageFill("SPY", legs, mids, 100, 3.0, True), open_flow)  # 3x costs worse
        mn, _ = PayoffExtremes(legs)
        max_loss = open_flow - mn * 100
        width = abs(legs[0]["K"] - legs[1]["K"]) * 100
        self.assertAlmostEqual(max_loss - open_flow, width, places=6)   # reserve == width x 100 (broker collateral)


class TestDayTradeGuard(unittest.TestCase):
    def setUp(self):
        self.days = [f"2026-11-{d:02d}" for d in (2, 3, 4, 5, 6, 9, 10, 11, 12, 13)]

    def test_overnight_not_day_trade(self):
        g = DayTradeGuard(self.days); g.RecordOpen("2026-11-02", "a", 1, 1)
        self.assertFalse(g.RecordClose("2026-11-03", "a", 1, 1)); self.assertEqual(g.CountInWindow("2026-11-03"), 0)

    def test_multileg_same_day_counts_per_leg_and_blocks(self):
        g = DayTradeGuard(self.days)
        for k in range(4): g.RecordOpen("2026-11-02", f"c:{k}", 7, 1)
        closes = [(f"c:{k}", 1) for k in range(4)]
        self.assertEqual(g.NewDayTrades("2026-11-02", closes), 4)
        self.assertFalse(g.CanClose("2026-11-02", closes))      # 4 > 3 worst case
        for k in range(3): g.RecordClose("2026-11-02", f"c:{k}", 7, 1)
        self.assertEqual(g.CountInWindow("2026-11-02"), 3); self.assertEqual(g.CountInWindow("2026-11-02", by="package"), 1)
        self.assertFalse(g.CanClose("2026-11-02", [("c:3", 1)]))

    def test_partial_close_and_staggered_lots(self):
        g = DayTradeGuard(self.days)
        g.RecordOpen("2026-11-02", "x", 1, 2); g.RecordOpen("2026-11-03", "x", 1, 2)
        self.assertTrue(g.RecordClose("2026-11-03", "x", 1, 1))   # same-day lot matched first (worst case)
        self.assertTrue(g.RecordClose("2026-11-03", "x", 1, 1))   # second partial also same-day
        self.assertFalse(g.RecordClose("2026-11-03", "x", 1, 2))  # remaining lot is from 11-02
        self.assertEqual(g.CountInWindow("2026-11-03"), 2)
        with self.assertRaises(ValueError):
            g.RecordClose("2026-11-04", "x", 1, 1)

    def test_roll_and_rolling_window_across_weeks(self):
        g = DayTradeGuard(self.days)
        for d in ("2026-11-02", "2026-11-03", "2026-11-04"):
            g.RecordOpen(d, f"r{d}", d, 1); g.RecordClose(d, f"r{d}", d, 1)
        g.RecordOpen("2026-11-05", "z", 9, 1)
        self.assertFalse(g.CanClose("2026-11-05", [("z", 1)]))   # would be 4th in rolling 5 sessions
        self.assertTrue(g.CanClose("2026-11-06", [("z", 1)]))    # holding overnight is fine
        # roll: close old leg (overnight) + open new leg same day = not a day trade
        g.RecordOpen("2026-11-05", "old", 5, 1)
        g.RecordOpen("2026-11-06", "new", 5, 1); self.assertFalse(g.RecordClose("2026-11-06", "old", 5, 1))
        self.assertEqual(g.CountInWindow("2026-11-06"), 3)
        self.assertEqual(g.CountInWindow("2026-11-09"), 2)   # 11-02 dropped out of 5-session window (11-03..11-09)
        self.assertEqual(g.CountInWindow("2026-11-11"), 0)


class TestVolModel(unittest.TestCase):
    def test_trailing_only(self):
        c = np.cumprod(1 + np.r_[0, np.random.default_rng(1).normal(0, 0.02, 300)]) * 100
        v1 = TrailingVol(c)["base"]; c2 = c.copy(); c2[200:] *= 1.5
        v2 = TrailingVol(c2)["base"]
        self.assertTrue(np.allclose(v1[:200], v2[:200], equal_nan=True))   # future data cannot change past estimates

    def test_skew_and_event(self):
        sc = IvScenario("t", 1.2, 0.1)
        self.assertGreater(ImpliedVol(0.3, 100, 90, 0.05, sc), ImpliedVol(0.3, 100, 110, 0.05, sc))
        self.assertGreater(ImpliedVol(0.3, 100, 100, 0.05, sc, event_var=0.01), ImpliedVol(0.3, 100, 100, 0.05, sc))


if __name__ == "__main__":
    unittest.main()
