"""PRE-REGISTERED event hypotheses (written 2026-11-09 ~16:10 ET, before any event-study result was computed).
Changing this file after results invalidates the pre-registration; the run logs its sha256."""
REACTION = "close(R-1)->close(R); R = first session whose close includes the release (BMO/intraday: event date; AMC: next session)"
HYPOTHESES = [
    {"id": "H1_fomc_preevent_long", "events": "FOMC", "tickers": ["SPY"], "kind": "underlying",
     "rule": "long SPY open(R-1) -> open(R+1) every scheduled FOMC (spans the 14:00 ET statement); no signal",
     "placebo": "same window on non-event sessions (same weekday, same year, >=3 sessions from any FOMC/CPI)"},
    {"id": "H2_fomc_reaction_continuation", "events": "FOMC", "tickers": ["SPY", "QQQ", "IWM", "TLT"], "kind": "underlying+synthetic_option",
     "rule": "if |reaction| > 1.0 x trailing daily sigma (known at close R-1), trade in reaction direction open(R+1) -> open(R+6); shares long-only (down -> cash); options: bull_call_debit / bear_put_debit ~2w expiry",
     "placebo": "same threshold rule on non-event sessions"},
    {"id": "H3_cpi_reaction_continuation", "events": "CPI", "tickers": ["SPY", "QQQ", "IWM", "TLT"], "kind": "underlying+synthetic_option",
     "rule": "as H2 but CPI (08:30 ET, before open)", "placebo": "same threshold rule on non-event sessions"},
    {"id": "H4_earnings_reaction_continuation", "events": "EARNINGS", "tickers": ["HOOD", "SHOP", "AXTI", "QURE"], "kind": "underlying+synthetic_option",
     "rule": "if |reaction| > 2.0 x trailing daily sigma, trade in reaction direction open(R+1) -> open(R+11) (does not cross next earnings); shares long-only; options debit spreads",
     "placebo": "same threshold on non-earnings sessions of the same ticker"},
    {"id": "H5_earnings_reaction_reversal", "events": "EARNINGS", "tickers": ["HOOD", "SHOP", "AXTI", "QURE"], "kind": "underlying+synthetic_option",
     "rule": "as H4 but trade AGAINST the reaction direction", "placebo": "same"},
    {"id": "H6_momentum_earnings_avoidance", "events": "EARNINGS", "tickers": "shares-momentum universe", "kind": "overlay",
     "rule": "shares momentum (all 10 weekday x parity phases) but drop any single name whose earnings reaction session falls inside the next holding period (known dates)",
     "placebo": "baseline = same phase without overlay"},
    {"id": "H7_momentum_macro_avoidance", "events": "FOMC+CPI", "tickers": "shares-momentum universe", "kind": "overlay",
     "rule": "shares momentum, go to cash at open(R-1) and re-enter at open(R+1) around every FOMC/CPI (all 10 phases)",
     "placebo": "baseline = same phase without overlay"},
    {"id": "H8_spy_event_condor_SYNTHETIC", "events": "FOMC+CPI", "tickers": ["SPY"], "kind": "synthetic_option_only",
     "rule": "sell SPY iron condor at open(R-1), buy back open(R+1); entry IV = trailing x IVRP x (1 + event bump), exit IV = trailing x IVRP; bump in {0, observed 2026 median pre-event IV excess (H10)}; no other invented premium",
     "placebo": "same structure on placebo sessions with bump 0"},
    {"id": "H9_earnings_straddle_RESEARCH_COMPARATOR", "events": "EARNINGS", "tickers": ["HOOD", "SHOP", "AXTI", "QURE"], "kind": "synthetic_option_only",
     "rule": "RESEARCH ONLY (crosses earnings; never a live candidate): long ATM straddle open(R-1)->open(R+1); entry event variance = mean |past earnings reaction|^2 (known at entry), exit without event var",
     "placebo": "none (diagnostic of the event-variance model)"},
    {"id": "H10_real_option_iv_around_events", "events": "FOMC+CPI+EARNINGS (2026 panel)", "tickers": "panel", "kind": "real_marks_descriptive",
     "rule": "change in inverted IV close(R-1)->close(R) for live panel contracts (DTE>3) vs non-event day pairs; descriptive only, RH bars are not bid/ask",
     "placebo": "non-event consecutive-session pairs, same contracts"},
]
EXCLUDED = [
    {"id": "X1_cpi_surprise", "why": "no free archived as-of consensus (pre-release survey median) for 2012-2026; today's or revised data cannot substitute"},
    {"id": "X2_fomc_surprise", "why": "no archived as-of fed-funds-futures implied path in the free/keyless sources used"},
    {"id": "X3_earnings_eps_surprise", "why": "Yahoo eps_estimate/eps_reported have unknown vintage (may be revised/backfilled); not an archived as-of consensus"},
    {"id": "X4_fda_policy_unscheduled", "why": "no complete timestamped archive with predetermined inclusion rules available; qualitative cases only (none added)"},
    {"id": "X5_implied_move_expectation", "why": "historical ATM straddle (implied move) only observable for the 2026 RH panel (H10); not available 2012-2025"},
]
SIZING = [0.01, 0.02, 0.05]
COSTS = [1.0, 2.0, 3.0]
IVRPS = [1.0, 1.1, 1.25]
SIGMA_MACRO, SIGMA_EARN = 1.0, 2.0
HOLD_MACRO, HOLD_EARN = 5, 10
