"""Regression tests for the v3 fill repair (2026-11-09 abstract-edge-search)."""
import unittest
from Options import Validate10k as V
from Options import SynthBacktest as SB


def spread(k1, k2, q1=1, q2=-1, kind="C"):
    return [{"kind": kind, "K": k1, "qty": q1}, {"kind": kind, "K": k2, "qty": q2}]


class FillRepair(unittest.TestCase):
    def setUp(self): self._fm, self._itm = V.FILL_MODEL, V.ITM_FRAC

    def tearDown(self): V.FILL_MODEL, V.ITM_FRAC = self._fm, self._itm

    def test_qure_deep_itm_close_defect_reproduced_and_fixed(self):
        legs = spread(9.0, 11.0); mids = [6.56, 4.57]; S = 15.55   # both legs deep ITM, width $2
        V.FILL_MODEL = "v2"; v2 = -V.Fill("QURE", legs, mids, 2.0, False, S)
        self.assertLess(v2, 100.0)   # documented defect: closing a $200-max package returns < half its value
        for itm in (0.0, 0.10, 0.25):
            V.FILL_MODEL, V.ITM_FRAC = "v3", itm
            v3 = -V.Fill("QURE", legs, mids, 2.0, False, S)
            self.assertGreater(v3, v2); self.assertGreaterEqual(v3, 0.0); self.assertLessEqual(v3, 200.0)
            self.assertLess(v3, (6.56 - 4.57) * 100)   # still worse than mid

    def test_otm_fills_unchanged(self):
        legs = spread(110.0, 115.0); mids = [1.20, 0.45]; S = 100.0
        V.FILL_MODEL = "v2"; a = V.Fill("SPY", legs, mids, 2.0, True, S); b = V.Fill("SPY", legs, mids, 2.0, False, S)
        V.FILL_MODEL = "v3"; self.assertAlmostEqual(a, V.Fill("SPY", legs, mids, 2.0, True, S), 9)
        self.assertAlmostEqual(b, V.Fill("SPY", legs, mids, 2.0, False, S), 9)

    def test_fees_and_friction_positive_for_original_and_inverse(self):
        V.FILL_MODEL = "v3"
        for sym, S, legs, mids in [("QURE", 15.55, spread(9.0, 11.0), [6.56, 4.57]), ("SPY", 500.0, spread(495.0, 505.0), [9.1, 3.2]),
                                   ("HOOD", 40.0, spread(30.0, 35.0, kind="P"), [0.20, 0.55])]:
            inv = [{**lg, "qty": -lg["qty"]} for lg in legs]
            for L in (legs, inv):
                mid_val = sum(lg["qty"] * m for lg, m in zip(L, mids)) * 100
                for cm in (1.0, 2.0, 3.0):
                    self.assertGreater(V.Fill(sym, L, mids, cm, True, S), mid_val)       # pay more than mid to open
                    self.assertLess(-V.Fill(sym, L, mids, cm, False, S), mid_val)      # receive less than mid to close
            rt = lambda L: V.Fill(sym, L, mids, 2.0, True, S) - V.Fill(sym, L, mids, 2.0, False, S) * -1
            self.assertGreater(rt(legs), 0); self.assertGreater(rt(inv), 0)

    def test_monotone_in_cost(self):
        V.FILL_MODEL = "v3"; legs = spread(9.0, 11.0); mids = [6.56, 4.57]
        c = [V.Fill("QURE", legs, mids, cm, True, 15.55) for cm in (0.5, 1, 2, 3)]
        self.assertEqual(c, sorted(c))
        p = [-V.Fill("QURE", legs, mids, cm, False, 15.55) for cm in (0.5, 1, 2, 3)]
        self.assertEqual(p, sorted(p, reverse=True))

    def test_exercise_floor_long_itm_put(self):
        V.FILL_MODEL = "v3"; legs = [{"kind": "P", "K": 60.0, "qty": 1}]
        got = -V.Fill("AXTI", legs, [19.70], 3.0, False, 40.0)   # European mid below intrinsic 20
        self.assertGreaterEqual(got, (20.0 - 3.0 * SB.STOCK_SLIP_BPS / 1e4 * 40.0) * 100 - V.Tick("AXTI", 19.9) * 100 - 3 * SB.FEE_PER_CONTRACT)
        self.assertLess(got, 2000.0)

    def test_inverse_structures(self):
        self.assertTrue(V.InverseAllowed("bull_call_debit")); self.assertTrue(V.InverseAllowed("iron_condor"))
        self.assertFalse(V.InverseAllowed("long_straddle")); self.assertFalse(V.InverseAllowed("long_call_atm"))
        self.assertFalse(V.InverseAllowed("covered_call"))
        legs = V.Legs("SPY", "inv:bull_call_debit", 500.0, 0.2, 14 / 365); orig = V.Legs("SPY", "bull_call_debit", 500.0, 0.2, 14 / 365)
        self.assertEqual([(l["K"], -l["qty"]) for l in legs], [(l["K"], l["qty"]) for l in orig])
        self.assertEqual(V.Bias("opp:bull_call_debit"), "bear"); self.assertEqual(V.Bias("inv:bull_call_debit"), "bull")


if __name__ == "__main__":
    unittest.main()
