"""Tests for the event phase: registry timing, no-lookahead entries, placebo exclusion, option settle bounds, bootstrap."""
import csv, math, unittest
import numpy as np
from Events import EventStudy as E
from Options.VolModel import IvScenario


class TestEvents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        E._init(); cls.D = E._D; cls.ev = cls.D["events"]; cls.dates = cls.D["dates"]

    def test_registry_fields_and_timing(self):
        with (E.ROOT / "data/events/registry.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for k in ("event_time_et", "known_at", "source_url", "expectation_asof", "actual_release_vintage"):
            self.assertIn(k, rows[0])
        for r in rows:
            if r["include"] != "True": continue
            self.assertGreaterEqual(r["reaction_session"], r["event_date"])          # never react before the release
            if r["event_type"] == "EARNINGS" and r["timing"] == "AMC":
                self.assertGreater(r["reaction_session"], r["event_date"])          # AMC -> next session at the earliest
            if r["event_type"] in ("CPI", "FOMC"):
                self.assertEqual(r["reaction_session"], r["event_date"])

    def test_scheduled_fomc_counts(self):
        from collections import Counter
        c = Counter(e["event_date"][:4] for e in self.ev if e["event_type"] == "FOMC")
        for y in range(2012, 2026):
            self.assertIn(c[str(y)], (7, 8), y)

    def test_placebo_excludes_events(self):
        Rs = sorted({e["R"] for e in self.ev if e["event_type"] in ("FOMC", "CPI")})
        P = set(E.Placebos(self.dates, Rs, 300, len(self.dates)))
        for r in Rs:
            for k in range(-3, 4): self.assertNotIn(r + k, P)

    def test_option_entry_after_reaction_and_settle_bounds(self):
        sc = IvScenario("t", 1.1, 0.1)
        e = next(x for x in self.ev if x["event_type"] == "CPI" and x["event_date"] >= "2019-01-01")
        R = e["R"]; x = E.OptTrade("SPY", "bull_call_debit", R + 1, R + 6, R + 11, sc, 2.0, settle=True)
        self.assertGreater(x["t_in"], R)
        width = max(float(l[3:]) for l in x["legs"].split(";")) - min(float(l[3:]) for l in x["legs"].split(";"))
        self.assertGreaterEqual(x["proc"], 0.0); self.assertLessEqual(x["proc"], width * 100 + 1e-6)
        self.assertGreaterEqual(x["pnl"], -x["max_loss"] - 1e-6)

    def test_cluster_bootstrap_ci_contains_mean(self):
        v = np.r_[np.ones(10), -np.ones(10)] * 0.01; cl = np.repeat(np.arange(10), 2)
        m, lo, hi, n, nc = E.ClusterBoot(v, cl, n=300)
        self.assertTrue(lo <= m <= hi); self.assertEqual(nc, 10)

    def test_account_no_lookahead_and_no_daytrades(self):
        sp = [{"sym": "SPY", "t_in": e["R"] + 1, "t_out": e["R"] + 6, "event_id": e["event_id"], "cm": 2.0} for e in self.ev if e["event_type"] == "CPI"]
        eq, done = E.Account(sp, "shares", 0.02)
        for p in done: self.assertGreater(p["t_out"], p["t_in"])
        self.assertTrue(np.all(eq[np.isfinite(eq)] > 0))


if __name__ == "__main__":
    unittest.main()
