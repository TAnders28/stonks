# Logged deviations from pre-registration (Events/Hypotheses.py sha256 dcdefff6...)
1. 2026-11-09 ~16:10 ET, after the first run (ev_20261109T160433): the synthetic option exit at the OPEN with the v2
   fill model charges HALF_SPREAD_PCT x full mid on deep-ITM legs (e.g. QURE 15% x $6.55 x 2 = $1.97 half-spread),
   which can make closing a debit spread cost more than its intrinsic value (ret -2.0x risk). Added a
   hold-to-expiry cash-settle variant (post-hoc, labeled) next to the pre-registered open exit for H2-H5, H8, H9.
   For H2/H3 the settle variant holds ~10 sessions instead of 5. Both are reported; neither replaces the other.
2. FOMC 2013-10-30 and 2014-03-19 force-marked scheduled (registry heuristic mis-flag); fixed before the first run.
