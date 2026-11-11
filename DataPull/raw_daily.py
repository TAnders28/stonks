"""Raw (contract-consistent) daily prices + corporate actions via keyless yfinance.

yfinance auto_adjust=False 'Close' is split-adjusted but NOT dividend-adjusted. We reconstruct the truly
unadjusted price that listed option strikes referenced at the time: raw = Close * prod(split ratios after date).
Writes data/daily_raw/<SYM>.json with bars (date, open, high, low, close [unadjusted], close_splitadj,
split_factor_after), dividends [(date, cash per share as-of-then)], splits [(date, ratio)].
Usage: .venv\\Scripts\\python -m DataPull.raw_daily --symbols SPY,QQQ,...
"""
from __future__ import annotations
import argparse, json, math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import yfinance as yf

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "daily_raw"


def Pull(sym):
    t = yf.Ticker(sym)
    df = t.history(period="max", interval="1d", auto_adjust=False, actions=True)
    splits = [(i.date().isoformat(), float(v)) for i, v in df["Stock Splits"].items() if v and v > 0]
    divs_sa = [(i.date().isoformat(), float(v)) for i, v in df["Dividends"].items() if v and v > 0]
    def factor_after(d):
        f = 1.0
        for sd, r in splits:
            if sd > d: f *= r
        return f
    bars = []
    for i, r in df.iterrows():
        d = i.date().isoformat(); f = factor_after(d)
        o, h, l, c = (float(r[k]) for k in ("Open", "High", "Low", "Close"))
        if not all(map(math.isfinite, (o, h, l, c))) or c <= 0: continue
        bars.append({"date": d, "open": o * f, "high": h * f, "low": l * f, "close": c * f, "close_splitadj": c,
                     "split_factor_after": f, "volume": float(r.get("Volume", 0) or 0)})
    divs = [(d, v * factor_after(d)) for d, v in divs_sa]
    return {"symbol": sym, "source": "yfinance auto_adjust=False + actions (keyless)", "fetched_at_et": datetime.now(ET).isoformat(timespec="seconds"),
            "basis": "UNADJUSTED (split factor re-applied); dividends in as-of-then cash", "splits": splits, "dividends": divs,
            "first": bars[0]["date"], "last": bars[-1]["date"], "bars": bars}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--symbols", default="SPY,QQQ,IWM,XBI,GDX,TLT,HOOD,AXTI,SHOP,QURE")
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    for s in a.symbols.split(","):
        js = Pull(s.strip()); (OUT / f"{s.strip()}.json").write_text(json.dumps(js))
        print(s, js["first"], js["last"], "splits", js["splits"][-3:], "n_div", len(js["dividends"]))


if __name__ == "__main__":
    main()
