"""Tests for the v2 $10k validation engine (ticks, raw/split handling, dividends, accounting, no-lookahead)."""
import math, unittest
import numpy as np
from Options import Validate10k as V
from Options import SynthBacktest as SB
from Options.VolModel import IvScenario
from Options.Pricing import PayoffExtremes


class TestTicks(unittest.TestCase):
    def test_tick_table(self):
        self.assertEqual(V.Tick("SPY", 5.0), 0.01); self.assertEqual(V.Tick("HOOD", 2.5), 0.01); self.assertEqual(V.Tick("HOOD", 3.5), 0.05)
        self.assertEqual(V.Tick("QURE", 1.0), 0.05); self.assertEqual(V.Tick("AXTI", 4.0), 0.10)

    def test_rounding_direction(self):
        self.assertAlmostEqual(V.RoundBuy(1.231, 0.05), 1.25); self.assertAlmostEqual(V.RoundSell(1.249, 0.05), 1.20)
        self.assertAlmostEqual(V.RoundBuy(1.25, 0.05), 1.25); self.assertEqual(V.RoundSell(0.004, 0.05), 0.0)

    def test_fill_on_tick_and_worse_than_mid(self):
        legs = [{"kind": "C", "K": 100, "qty": 1}, {"kind": "C", "K": 105, "qty": -1}]
        mids = [3.333, 1.111]
        pay = V.Fill("AXTI", legs, mids, 1.0, True) - 2 * SB.FEE_PER_CONTRACT
        self.assertGreater(pay, (3.333 - 1.111) * 100)
        self.assertAlmostEqual((pay / 100) / 0.05, round((pay / 100) / 0.05), places=6)  # both legs on 0.05/0.10 grid
        rec = -V.Fill("AXTI", legs, mids, 1.0, False) + 2 * SB.FEE_PER_CONTRACT
        self.assertLess(rec, (3.333 - 1.111) * 100)


class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        V._init(); cls.D = V._D

    def test_raw_vs_adjusted(self):
        tk = self.D["tick"]["SHOP"]; d = self.D["dates"]; i = d.index("2022-06-28"); j = d.index("2022-06-29")
        self.assertGreater(tk["Cr"][i] / tk["Cr"][j], 5)          # unadjusted 10:1 split visible in raw
        self.assertLess(abs(tk["C"][i] / tk["C"][j] - 1), 0.25)     # adjusted series has no jump
        spy = self.D["tick"]["SPY"]; k = d.index("2013-01-02")
        self.assertGreater(spy["Cr"][k], spy["C"][k])             # dividend adjustment lowers old adjusted prices

    def test_split_cycles_excluded(self):
        r = V.RunOptions2(V.Cfg2("long_call_atm", "expiry", "r02c10", V.SCENARIOS[1], 1.0))
        self.assertGreaterEqual(r["stats"]["split_excluded"], 0)
        r2 = V.RunOptions2(V.Cfg2("iron_condor", "expiry", "r02c10", V.SCENARIOS[1], 1.0))
        self.assertGreater(r2["stats"]["split_excluded"], 0)       # any-bias structure must hit XBI 2015 / SHOP 2022 cycles

    def test_accounting_reconciles_and_zero_day_trades(self):
        for s, x in (("covered_call", "expiry"), ("iron_condor", "tminus1"), ("bull_call_debit", "expiry")):
            r = V.RunOptions2(V.Cfg2(s, x, "r05c25", V.SCENARIOS[1], 2.0))
            self.assertLess(abs(r["recon_gap"]), 1.0, s); self.assertEqual(r["day_trades_leg"], 0)
            self.assertGreaterEqual(r["min_equity"], 0.0)

    def test_dividends_credited_to_covered_calls(self):
        # at $10k a 100-share covered call fits the 1-5% risk grid only for sub-$5 names (no dividends), so test
        # the dividend path with a test-only wide sizing key
        V.SIZING["_test_wide"] = (0.5, 1.0)
        try:
            r = V.RunOptions2(V.Cfg2("covered_call", "expiry", "_test_wide", V.SCENARIOS[1], 1.0))
        finally:
            V.SIZING.pop("_test_wide")
        self.assertGreater(r["stats"]["div_credits"], 0); self.assertLess(abs(r["recon_gap"]), 1.0)

    def test_no_lookahead_entry_inputs(self):
        # perturbing data AFTER the first entry must not change that entry's package
        import copy
        cfg = V.Cfg2("long_call_atm", "tminus1", "r05c25", V.SCENARIOS[1], 1.0, record=True)
        a = V.RunOptions2(cfg, self.D)["trades"]; fa = a[0]; sym = fa["symbol"]
        t = self.D["dates"].index(fa["entry_date"])
        D2 = copy.deepcopy(self.D); tk = D2["tick"][sym]
        for arr in ("Or", "Cr", "C", "Cff"): tk[arr][t + 1:] *= 1.5
        tk["base"][t:] *= 3
        b = V.RunOptions2(cfg, D2)["trades"]
        fb = [x for x in b if x["symbol"] == sym and x["entry_date"] == fa["entry_date"]][0]
        self.assertEqual(fa["entry_date"], fb["entry_date"]); self.assertEqual(fa["entry_cost_pkg"], fb["entry_cost_pkg"]); self.assertEqual(fa["legs"], fb["legs"])

    def test_no_unbounded_structures(self):
        for s in V.STRUCTS:
            legs = V.Legs("SPY", s, 500.0, 0.2, 10 / 365)
            self.assertTrue(math.isfinite(PayoffExtremes(legs)[0]), s)


if __name__ == "__main__":
    unittest.main()
