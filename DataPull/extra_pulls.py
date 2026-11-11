"""Extra keyless pulls for the abstract-edge search: CBOE VIX/VIX9D (yfinance) and SEC EDGAR submissions (8-K/6-K acceptance times)."""
import json, time, requests, yfinance as yf
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ET=ZoneInfo("America/New_York"); R=Path("data"); (R/"vol_index").mkdir(exist_ok=True); (R/"edgar").mkdir(exist_ok=True)
for s in []:  # VIX already pulled
    df=yf.Ticker(s).history(period="max",interval="1d",auto_adjust=False)
    bars=[{"date":i.date().isoformat(),"open":float(r.Open),"close":float(r.Close)} for i,r in df.iterrows() if r.Close>0]
    (R/"vol_index"/f"{s.strip('^')}.json").write_text(json.dumps({"symbol":s,"source":"yfinance (CBOE index, keyless)","fetched_at_et":datetime.now(ET).isoformat(timespec="seconds"),"bars":bars}))
    print(s,bars[0]["date"],bars[-1]["date"],len(bars))
UA={"User-Agent":"stonk-data local research script (personal, non-commercial) admin@localhost","Accept-Encoding":"gzip, deflate"}
m=requests.get("https://www.sec.gov/files/company_tickers.json",headers=UA,timeout=30); print("tickers",m.status_code)
cik={"HOOD":1783879,"SHOP":1594805,"AXTI":1051627,"QURE":1590560}  # SEC CIKs (company_tickers.json returned 403)
for t in ["HOOD","SHOP","AXTI","QURE"]:
    c=cik.get(t); print(t,c)
    if not c: continue
    j=requests.get(f"https://data.sec.gov/submissions/CIK{c:010d}.json",headers=UA,timeout=30); time.sleep(0.3)
    if not j.ok: print(t,"status",j.status_code); continue
    js=j.json(); allf=[js["filings"]["recent"]]
    for f in js["filings"].get("files",[]):
        k=requests.get("https://data.sec.gov/submissions/"+f["name"],headers=UA,timeout=30); time.sleep(0.3)
        if k.ok: allf.append(k.json())
    rows=[]
    for b in allf:
        for i in range(len(b["form"])):
            if b["form"][i] in ("8-K","6-K","8-K/A"):
                rows.append({"form":b["form"][i],"filingDate":b["filingDate"][i],"acceptanceDateTime":b["acceptanceDateTime"][i],"items":b.get("items",[""]*len(b["form"]))[i],"accession":b["accessionNumber"][i],"primaryDocument":b["primaryDocument"][i],"primaryDocDescription":b.get("primaryDocDescription",[""]*len(b["form"]))[i]})
    (R/"edgar"/f"{t}.json").write_text(json.dumps({"ticker":t,"cik":c,"source":"data.sec.gov submissions (keyless)","fetched_at_et":datetime.now(ET).isoformat(timespec="seconds"),"filings":rows}))
    print(t,len(rows),"8-K/6-K; with 2.02:",sum("2.02" in r["items"] for r in rows), rows[0] if rows else None)

