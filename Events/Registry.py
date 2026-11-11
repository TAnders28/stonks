"""Event registry: scheduled FOMC statements (federalreserve.gov), CPI releases (BLS archive via Wayback copy of the
official page), earnings (Yahoo timestamps, timing verified against price reaction). Writes data/events/registry.csv.
Usage: .venv\\Scripts\\python -m Events.Registry
"""
from __future__ import annotations
import csv, json, re
from datetime import date
from pathlib import Path
import numpy as np, requests

ROOT = Path(__file__).resolve().parents[1]
OUTD = ROOT / "data" / "events"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
FED = ["https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"] + [f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm" for y in range(2011, 2021)]


def Sessions():
    js = json.loads((ROOT / "data/daily/SPY.json").read_text()); return [b["date"] for b in js["bars"]]


def Fomc():
    out = {}
    raw = OUTD / "fed_raw"; raw.mkdir(parents=True, exist_ok=True)
    for u in FED:
        h = requests.get(u, headers=UA, timeout=40).text; (raw / u.rsplit("/", 1)[1]).write_text(h, encoding="utf-8")
        for m in re.finditer(r'href="(/newsevents/press(?:releases/monetary|/monetary/)(\d{8})a\.htm)"', h):
            d = m.group(2); iso = f"{d[:4]}-{d[4:6]}-{d[6:]}"
            ctx = h[max(0, m.start() - 1500):m.start()].lower()
            last_panel = ctx.rfind("fomc-meeting__month")  # current-calendar markup
            unsched = "unscheduled" in ctx[ctx.rfind("<strong") if ctx.rfind("<strong") > 0 else 0:] or "notation vote" in ctx[-400:]
            out.setdefault(iso, {"date": iso, "source_url": "https://www.federalreserve.gov" + m.group(1), "calendar_page": u, "unscheduled_flag": unsched})
    return sorted(out.values(), key=lambda r: r["date"])


def main():
    OUTD.mkdir(parents=True, exist_ok=True); S = Sessions(); si = {d: i for i, d in enumerate(S)}
    def sess_on_or_after(d):
        for x in S:
            if x >= d: return x
    rows = []
    # manual override (documented): heuristic mis-flags these SCHEDULED meetings (Oct 29-30 2013, Mar 18-19 2014)
    FORCE_SCHEDULED = {"2013-10-30", "2014-03-19"}
    for r in Fomc():
        if r["date"] in FORCE_SCHEDULED: r["unscheduled_flag"] = False
        if not ("2012-01-01" <= r["date"] <= "2026-11-06"): continue
        y = int(r["date"][:4])
        rows.append({"event_id": f"FOMC_{r['date']}", "event_type": "FOMC", "symbol": "MACRO", "event_date": r["date"],
                     "event_time_et": f"{r['date']}T14:00" if y >= 2013 else f"{r['date']}T12:30/14:15 (2012 varied; intraday)",
                     "timing": "INTRADAY", "known_at": "meeting date on FOMC calendar published ~1y ahead (scheduled)",
                     "reaction_session": r["date"] if r["date"] in si else sess_on_or_after(r["date"]),
                     "source_url": r["source_url"], "expectation_asof": "NONE (no archived as-of fed funds futures path)",
                     "actual_release_vintage": "statement as first published (not used as a signal value)",
                     "scheduled": not r["unscheduled_flag"], "timing_check": "n/a", "include": (not r["unscheduled_flag"]) and r["date"] in si})
    with (ROOT / "data/events_cpi_bls.csv").open() as f:
        for r in csv.DictReader(f):
            d = r["release_date"]
            if not ("2012-01-01" <= d <= "2026-11-06"): continue
            rows.append({"event_id": f"CPI_{d}", "event_type": "CPI", "symbol": "MACRO", "event_date": d, "event_time_et": f"{d}T08:30",
                         "timing": "BMO", "known_at": "BLS release calendar published in advance", "reaction_session": d if d in si else sess_on_or_after(d),
                         "source_url": r["source_url"] + " (list via web.archive.org snapshot 2026-11-01 of https://www.bls.gov/bls/news-release/cpi.htm)",
                         "expectation_asof": "NONE (no archived consensus)", "actual_release_vintage": f"first release for {r['reference_month']} (not used as a signal value)",
                         "scheduled": True, "timing_check": "n/a", "include": d in si})
    # earnings: timing verification vs adjusted price reaction
    for s in ["HOOD", "SHOP", "AXTI", "QURE"]:
        ej = json.loads((ROOT / f"data/earnings/{s}.json").read_text())
        px = {b["date"]: b["close"] for b in json.loads((ROOT / f"data/daily/{s}.json").read_text())["bars"]}
        ds = [d for d in S if d in px]; di = {d: i for i, d in enumerate(ds)}; c = np.array([px[d] for d in ds]); lr = np.r_[np.nan, np.diff(np.log(c))]
        for e in ej["dates"]:
            d = e["date"]
            if not ("2012-01-01" <= d <= "2026-11-06") or e.get("eps_reported") is None and d > "2026-11-06": continue
            base = sess_on_or_after(d)
            if base is None or base not in di: continue
            k = di[base]; exp_k = k if e["timing"] == "BMO" else k + 1
            if exp_k + 1 >= len(ds): continue
            cand = {j: abs(lr[j]) for j in (k, k + 1) if j < len(ds)}
            argmax = max(cand, key=cand.get); ok = argmax == exp_k
            # conservative: unverified timing -> treat reaction as one session later (wait one full session)
            R = exp_k if ok else max(k + 1, exp_k)
            rows.append({"event_id": f"EARN_{s}_{d}", "event_type": "EARNINGS", "symbol": s, "event_date": d, "event_time_et": e["earnings_at_et"],
                         "timing": e["timing"], "known_at": "date announced by company before release (Yahoo calendar; as-of announcement date not archived)",
                         "reaction_session": ds[R], "source_url": "yfinance get_earnings_dates (Yahoo) fetched 2026-11-09 15:26 ET",
                         "expectation_asof": "EXCLUDED (Yahoo eps_estimate vintage unknown)", "actual_release_vintage": "unknown vintage (not used)",
                         "scheduled": True, "timing_check": "verified" if ok else "UNVERIFIED->shifted_to_later_session", "include": True})
    cols = list(rows[0].keys())
    with (OUTD / "registry.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in rows]
    from collections import Counter
    print(Counter((r["event_type"], r["include"]) for r in rows))
    print("FOMC per year", Counter(r["event_date"][:4] for r in rows if r["event_type"] == "FOMC" and r["include"]))
    print("FOMC flagged unscheduled", [r["event_date"] for r in rows if r["event_type"] == "FOMC" and not r["scheduled"]])
    print("earn timing", Counter((r["symbol"], r["timing_check"]) for r in rows if r["event_type"] == "EARNINGS"))


if __name__ == "__main__":
    main()
