"""Targeted leakage/accounting tests for Adaptive.Policy (2026-11-09)."""
import copy, math, unittest
import numpy as np
from Adaptive import Policy as AP
from Options import SynthBacktest as SB

_D = {}


def Data():
    if not _D: _D["D"] = AP.LoadAll()
    return _D["D"]


def Perturb(D, j, f=1.3):
    """Copy with every price/vol series multiplied by f strictly after index j (future), features recomputed."""
    E = dict(D); E["tick"] = {s: dict(tk) for s, tk in D["tick"].items()}
    for tk in E["tick"].values():
        for key in ("O", "C", "Cff", "Or", "Cr"):
            a = tk[key].copy(); a[j + 1:] *= f; tk[key] = a
        b = tk["base"].copy(); b[j + 1:] *= f; tk["base"] = b
    for key in ("vix", "vix9d"):
        a = D[key].copy(); a[j + 1:] *= f; E[key] = a
    E["F"] = AP.Features(E); return E


class Adaptive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.D = Data(); cls.sc = AP.Scen("t")
        cls.PS = AP.Periods(cls.D, 0)
        cls.PK = [{a: AP.Package(cls.D, cls.sc, P, a) for a in AP.OPT_ARMS} for P in cls.PS]
        cls.L, cls.known = AP.Labels(cls.D, cls.sc, cls.PS, cls.PK)
        cls.K0 = next(j for j in range(len(cls.PS)) if len(AP.Usable(cls.PS, cls.known, j)) >= AP.MIN_TRAIN)

    def test_state_uses_only_prior_close(self):
        t = self.PS[200]["t"]; s0 = AP.StateAt(self.D, t)
        E = Perturb(self.D, t - 1, 2.0)          # change day t (open AND close) and everything after
        self.assertEqual(s0, AP.StateAt(E, t))

    def test_labels_only_after_realized_with_embargo(self):
        for j in range(self.K0, len(self.PS), 37):
            ks = AP.Usable(self.PS, self.known, j)
            self.assertTrue(all(k <= j - 2 for k in ks))
            self.assertTrue(all(self.known[k].max() < self.PS[j]["t"] for k in ks))
            self.assertGreaterEqual(min(self.PS[j]["t"] - self.known[k].max() for k in ks), 1)

    def test_future_changes_do_not_alter_past_decisions(self):
        j = (self.K0 + len(self.PS)) // 2; tj = self.PS[j]["t"]
        E = Perturb(self.D, tj - 1, 1.5)
        PK2 = [{a: AP.Package(E, self.sc, P, a) for a in AP.OPT_ARMS} for P in self.PS]
        L2, kn2 = AP.Labels(E, self.sc, self.PS, PK2)
        a1, _ = AP.PolicyRidge(self.D, self.PS, self.L, self.known, self.K0); a2, _ = AP.PolicyRidge(E, self.PS, L2, kn2, self.K0)
        b1, _ = AP.PolicyTable(self.D, self.PS, self.K0); b2, _ = AP.PolicyTable(E, self.PS, self.K0)
        n = j - self.K0 + 1     # decisions up to and including t_j (state from close t_j-1, labels known < t_j)
        self.assertEqual(a1[:n], a2[:n]); self.assertEqual(b1[:n], b2[:n])
        self.assertNotEqual(a1 + b1, a2 + b2)   # sanity: perturbation is not a no-op later

    def test_capital_reconciles_no_daytrades_whole_contracts(self):
        al, _ = AP.PolicyRidge(self.D, self.PS, self.L, self.known, self.K0)
        for name, alloc in (("B", al), ("ic", [{"iron_condor": 1.0}] * (len(self.PS) - self.K0))):
            r = AP.Simulate(self.D, self.sc, self.PS, self.PK, self.K0, alloc, name, record=True)
            self.assertLess(abs(r["led"]["recon_gap"]), 0.01, r["led"])
            self.assertEqual(r["led"]["day_trades"], 0)
            for tr in r["trades"]:
                if tr["action"] == "open": self.assertIsInstance(tr["qty"], int); self.assertGreaterEqual(tr["qty"], 1)
            self.assertGreater(min(r["eq"]), 0)

    def test_open_risk_within_budget(self):
        k = self.K0 + 5; pk = self.PK[k]["bull_call_debit"]
        r = AP.Simulate(self.D, self.sc, self.PS, self.PK, k, [{"bull_call_debit": 1.0}] * (len(self.PS) - k), "x", record=True)
        tr = r["trades"][0]; self.assertLessEqual(tr["qty"] * pk["max_loss"], self.sc.B * AP.ACC + 1e-6)

    def test_transition_pays_close_and_open_costs(self):
        k = self.K0; alloc = [{"equity": 1.0}, {"bull_call_debit": 1.0}, {"cash": 1.0}]
        PS = self.PS[:k + 4]
        r = AP.Simulate(self.D, self.sc, PS, self.PK, k, alloc + [{"cash": 1.0}], "tr", record=True)
        d2 = self.D["dates"][PS[k + 1]["t"]]
        acts = [(x["action"], x["instrument"]) for x in r["trades"] if x["date"] == d2]
        self.assertIn(("open", "bull_call_debit"), acts); self.assertTrue(any(a == "sell" for a, _ in acts))
        self.assertTrue(all(x["pnl"] < 0 for x in r["trades"] if x["date"] == d2 and x["action"] == "sell"))
        self.assertGreater(r["led"]["opt_spread_cost"], 0); self.assertGreater(r["led"]["opt_fees"], 0)
        self.assertTrue(any(x["action"] == "close" for x in r["trades"]))

    def test_central_iv_is_observed_vix9d_not_invented(self):
        P = self.PS[self.K0]; t = P["t"]
        self.assertEqual(AP.BaseVol(self.D, self.sc, t - 1), self.D["vix9d"][t - 1] / 100.0)
        self.assertTrue(np.isfinite(self.D["vix9d"][self.PS[0]["t"] - 1]))

    def test_zero_cost_still_tick_rounds(self):
        sc0 = AP.Scen("z", cost_mult=0.0); pk = AP.Package(self.D, sc0, self.PS[self.K0], "iron_condor")
        self.assertGreaterEqual(pk["cost"], pk["mid_open"] - 1e-9)   # credit received never better than mid


if __name__ == "__main__":
    unittest.main()
