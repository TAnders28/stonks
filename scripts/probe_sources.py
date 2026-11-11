"""Probe keyless stock data sources. Writes JSON report to out/."""
from __future__ import annotations
import json, time, traceback
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
TICKERS = ["HOOD", "AXTI", "SHOP", "QURE", "SPY"]
OUT = Path("/workspace/stonk-data/out")
CACHE = Path("/workspace/stonk-data/cache/raw")
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

def now_et():
    return datetime.now(ET).isoformat(timespec="seconds")

def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

report = {"fetched_at_et": now_et(), "tickers": TICKERS, "sources": {}}

# ---- yfinance ----
yf_block = {"ok": False}
try:
    import yfinance as yf
    t0 = time.time()
    quotes = {}
    for sym in TICKERS:
        t = yf.Ticker(sym)
        info = {}
        try:
            fi = t.fast_info
            info = {
                "last_price": float(getattr(fi, "last_price", None) or getattr(fi, "lastPrice", None) or 0) or None,
                "previous_close": float(getattr(fi, "previous_close", None) or getattr(fi, "previousClose", None) or 0) or None,
                "last_volume": int(getattr(fi, "last_volume", None) or getattr(fi, "lastVolume", None) or 0) or None,
                "timezone": str(getattr(fi, "timezone", None) or ""),
            }
        except Exception as e:
            info = {"fast_info_error": str(e)}
        quotes[sym] = info
    # daily history 5y SPY + HOOD
    hist = {}
    for sym in ["SPY", "HOOD", "AXTI", "SHOP", "QURE"]:
        df = yf.download(sym, period="5y", interval="1d", auto_adjust=True, progress=False, threads=False)
        if isinstance(df.columns, type(df.columns)) and getattr(df.columns, "nlevels", 1) > 1:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        hist[sym] = {
            "rows": int(len(df)),
            "start": str(df.index.min()) if len(df) else None,
            "end": str(df.index.max()) if len(df) else None,
            "columns": list(map(str, df.columns)),
            "tail": df.tail(2).reset_index().astype(str).to_dict(orient="records") if len(df) else [],
        }
        # cache raw jsonl-ish
        if len(df):
            records = []
            for idx, row in df.iterrows():
                records.append({
                    "date": str(idx.date()) if hasattr(idx, "date") else str(idx)[:10],
                    "open": float(row.get("Open", row.iloc[0])),
                    "high": float(row.get("High", row.iloc[1])),
                    "low": float(row.get("Low", row.iloc[2])),
                    "close": float(row.get("Close", row.iloc[3])),
                    "volume": float(row.get("Volume", row.iloc[4]) if "Volume" in df.columns else 0),
                })
            p = CACHE / f"yfinance_daily_{sym}_{stamp()}.json"
            p.write_text(json.dumps({"source":"yfinance","symbol":sym,"auto_adjust":True,
                                     "fetched_at_et": now_et(), "bars": records}, indent=2))
    # intraday 1m / 5m (limited lookback)
    intra = {}
    for interval, period in [("1m","5d"), ("5m","60d"), ("1h","2y")]:
        try:
            df = yf.download("SPY", period=period, interval=interval, auto_adjust=True, progress=False, threads=False)
            if getattr(df.columns, "nlevels", 1) > 1:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            intra[interval] = {"rows": int(len(df)), "start": str(df.index.min()) if len(df) else None,
                               "end": str(df.index.max()) if len(df) else None, "period_requested": period}
        except Exception as e:
            intra[interval] = {"error": str(e)}
    # option chain snapshot (live only, not historical)
    opt = {}
    try:
        tk = yf.Ticker("HOOD")
        exps = list(tk.options)[:3]
        opt["expirations_sample"] = exps
        if exps:
            chain = tk.option_chain(exps[0])
            opt["calls_rows"] = int(len(chain.calls))
            opt["puts_rows"] = int(len(chain.puts))
            opt["call_cols"] = list(chain.calls.columns)
            opt["sample_call"] = chain.calls.head(1).astype(str).to_dict(orient="records")
    except Exception as e:
        opt["error"] = str(e)
    yf_block = {
        "ok": True,
        "latency_s": round(time.time()-t0, 2),
        "quotes_near_live": quotes,
        "daily_5y": hist,
        "intraday": intra,
        "option_chain_live": opt,
        "notes": "Yahoo-sourced via yfinance; near-live quotes often delayed ~15m for free tier; auto_adjust=True = split+div adjusted closes.",
    }
except Exception as e:
    yf_block = {"ok": False, "error": str(e), "trace": traceback.format_exc()[-800:]}
report["sources"]["yfinance"] = yf_block

# ---- Stooq direct CSV ----
stooq = {"ok": False}
try:
    import urllib.request
    t0 = time.time()
    results = {}
    for sym in TICKERS:
        # US stocks on Stooq use .us suffix lowercase
        url = f"https://stooq.com/q/d/l/?s={sym.lower()}.us&i=d"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            lines = [ln for ln in raw.strip().splitlines() if ln.strip()]
            header = lines[0] if lines else ""
            n = max(0, len(lines)-1)
            sample = lines[1:3] + lines[-2:] if n >= 2 else lines[1:]
            results[sym] = {"url": url, "header": header, "rows": n, "sample_lines": sample,
                            "http_ok": True}
            (CACHE / f"stooq_daily_{sym}_{stamp()}.csv").write_text(raw)
        except Exception as e:
            results[sym] = {"url": url, "error": str(e)}
    # try intraday? Stooq d/l is daily only typically
    stooq = {"ok": any(r.get("rows",0)>0 for r in results.values()),
             "latency_s": round(time.time()-t0, 2),
             "tickers": results,
             "notes": "Direct CSV download, no key. Daily only (i=d). Adjusted? Stooq usually adjusted close in Close column for .us."}
except Exception as e:
    stooq = {"ok": False, "error": str(e), "trace": traceback.format_exc()[-500:]}
report["sources"]["stooq_csv"] = stooq

# ---- pandas-datareader Stooq ----
pdr = {"ok": False}
try:
    import pandas_datareader.data as web
    t0 = time.time()
    df = web.DataReader("SPY.US", "stooq")  # newest first typically
    pdr = {
        "ok": True,
        "latency_s": round(time.time()-t0, 2),
        "rows": int(len(df)),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "columns": list(map(str, df.columns)),
        "tail": df.sort_index().tail(2).reset_index().astype(str).to_dict(orient="records"),
    }
except Exception as e:
    pdr = {"ok": False, "error": str(e), "trace": traceback.format_exc()[-500:]}
report["sources"]["pandas_datareader_stooq"] = pdr

# ---- Yahoo chart API direct (no yfinance) ----
ychart = {"ok": False}
try:
    import urllib.request
    t0 = time.time()
    url = "https://query1.finance.yahoo.com/v8/finance/chart/SPY?range=5d&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode()
    data = json.loads(body)
    meta = data.get("chart",{}).get("result",[{}])[0].get("meta",{})
    ts = data.get("chart",{}).get("result",[{}])[0].get("timestamp",[])
    ychart = {
        "ok": True,
        "latency_s": round(time.time()-t0, 2),
        "meta_keys": list(meta.keys())[:20],
        "regularMarketPrice": meta.get("regularMarketPrice"),
        "regularMarketTime": meta.get("regularMarketTime"),
        "exchangeTimezoneName": meta.get("exchangeTimezoneName"),
        "bars": len(ts or []),
        "notes": "Unofficial Yahoo chart endpoint; same backend as yfinance; no key but ToS/rate-limit risk.",
    }
    (CACHE / f"yahoo_chart_SPY_{stamp()}.json").write_text(body[:200000])
except Exception as e:
    ychart = {"ok": False, "error": str(e)}
report["sources"]["yahoo_chart_direct"] = ychart

# ---- CNBC / other quick try: Nasdaq public API style ----
alt = {}
try:
    import urllib.request
    url = "https://api.nasdaq.com/api/quote/SPY/info?assetclass=etf"
    req = urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0","Accept":"application/json"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        body = resp.read().decode()
    j = json.loads(body)
    alt["nasdaq_quote"] = {"ok": True, "keys": list(j.keys()), "sample": str(j)[:400]}
except Exception as e:
    alt["nasdaq_quote"] = {"ok": False, "error": str(e)}
report["sources"]["nasdaq_public"] = alt.get("nasdaq_quote", {})

out_path = OUT / f"source_probe_{stamp()}.json"
out_path.write_text(json.dumps(report, indent=2, default=str))
print("WROTE", out_path)
# concise stdout summary
for name, blk in report["sources"].items():
    print(f"\n== {name} ok={blk.get('ok')} ==")
    print({k:blk[k] for k in blk if k not in ("trace",) and not isinstance(blk[k], (dict,list)) or k in ("notes","latency_s","error","ok")})
    if name=="yfinance" and blk.get("ok"):
        for s,h in blk["daily_5y"].items():
            print(f"  daily {s}: rows={h['rows']} {h['start']} -> {h['end']}")
        print("  intraday:", blk["intraday"])
        print("  options:", {k:blk['option_chain_live'].get(k) for k in ('expirations_sample','calls_rows','puts_rows','error')})
    if name=="stooq_csv" and blk.get("tickers"):
        for s,r in blk["tickers"].items():
            print(f"  {s}: rows={r.get('rows')} err={r.get('error')}")
