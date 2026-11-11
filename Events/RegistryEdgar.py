"""Clean earnings timing from contemporaneous SEC EDGAR acceptance timestamps (replaces the outcome-selected
reaction-argmax step in Events/Registry.py). Reads data/events/registry.csv (FOMC/CPI rows kept unchanged) and
data/edgar/<SYM>.json (data.sec.gov submissions, fetched 2026-11-09). Writes data/events/registry_edgar.csv.
Rule (frozen before any rerun): for each Yahoo earnings date d, take the EARLIEST 8-K whose items include 2.02 accepted
between d-1 00:00 ET and d+1 23:59 ET. Acceptance ET < 09:30 -> reaction session = first session on/after that date;
09:30-16:00 -> intraday: reaction session = that session (signal only usable from its close; entries are next open);
>= 16:00 -> next session. An 8-K is accepted at or after the press release, so this can only classify LATER, never
earlier, than the true release (no lookahead). No match, or 6-K only (no item codes) -> excluded (6-K date-match kept
as a labelled sensitivity set, include=False). Price data is NEVER consulted.
"""
from __future__ import annotations
import csv, json
from datetime import datetime, timedelta, date
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
ET = ZoneInfo("America/New_York")


def Sessions():
    return [b["date"] for b in json.loads((ROOT / "data/daily/SPY.json").read_text())["bars"]]


def main():
    S = Sessions()
    def on_or_after(d):
        for x in S:
            if x >= d: return x
    rows = list(csv.DictReader((ROOT / "data/events/registry.csv").open(encoding="utf-8")))
    out = [r for r in rows if r["event_type"] != "EARNINGS"]
    cov = {}
    for r in [r for r in rows if r["event_type"] == "EARNINGS"]:
        s, d = r["symbol"], r["event_date"]
        ed = json.loads((ROOT / f"data/edgar/{s}.json").read_text())
        fil = ed["filings"]; earliest = min(f["filingDate"] for f in fil); cov[s] = earliest
        lo = datetime.fromisoformat(d).replace(tzinfo=ET) - timedelta(days=1); hi = datetime.fromisoformat(d).replace(tzinfo=ET) + timedelta(days=2)
        def acc(f): return datetime.fromisoformat(f["acceptanceDateTime"].replace("Z", "+00:00")).astimezone(ET)
        m8 = sorted([f for f in fil if f["form"].startswith("8-K") and "2.02" in (f.get("items") or "") and lo <= acc(f) < hi], key=acc)
        m6 = sorted([f for f in fil if f["form"].startswith("6-K") and lo <= acc(f) < hi], key=acc)
        nr = dict(r); nr["yahoo_timing"] = r["timing"]; nr["old_reaction_session"] = r["reaction_session"]
        if m8 or m6:
            f = (m8 or m6)[0]; a = acc(f); ad = a.date().isoformat(); hm = a.hour * 60 + a.minute
            if hm < 9 * 60 + 30: tim, rs = "BMO", on_or_after(ad)
            elif hm < 16 * 60: tim, rs = "INTRADAY", on_or_after(ad)
            else: tim, rs = "AMC", on_or_after((a.date() + timedelta(days=1)).isoformat())
            nr.update({"timing": tim, "event_time_et": a.strftime("%Y-%m-%dT%H:%M ET (SEC acceptance)"), "reaction_session": rs,
                       "source_url": f"https://www.sec.gov/Archives/edgar/data/{ed['cik']}/{f['accession'].replace('-', '')}/{f['primaryDocument']}",
                       "known_at": "SEC EDGAR acceptanceDateTime (contemporaneous, independent of price)",
                       "timing_check": "edgar_8k_2.02" if m8 else "edgar_6k_datematch_item_unverified", "include": bool(m8)})
        else:
            nr.update({"timing_check": "no_edgar_match_excluded" if d >= earliest else "before_edgar_coverage_excluded", "include": False})
        out.append(nr)
    cols = list(dict.fromkeys(k for r in out for k in r))
    with (ROOT / "data/events/registry_edgar.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval=""); w.writeheader(); [w.writerow(r) for r in out]
    from collections import Counter
    E = [r for r in out if r["event_type"] == "EARNINGS"]
    print("edgar coverage from", cov)
    print(Counter((r["symbol"], r["timing_check"]) for r in E))
    print("timing changed vs old reaction session:", sum(1 for r in E if r["include"] in (True, "True") and r["reaction_session"] != r["old_reaction_session"]),
          "of", sum(1 for r in E if r["include"] in (True, "True")))
    print("yahoo vs edgar timing", Counter((r["yahoo_timing"], r["timing"]) for r in E if r["include"] in (True, "True")))


if __name__ == "__main__":
    main()
