"""Outcome check (NOT calibration): observed IV vs REALIZED vol over the option's remaining life (uses future
bars on purpose, as the outcome). Tests whether real 2026 options carried a variance risk premium (IV > later RV)."""
import pandas as pd, numpy as np, json, sys
D=sys.argv[1]; b="2026-11-09_options-validation"
d=pd.read_csv(f"{D}/{b}_realhist-iv.csv"); d=d[d.days_to_exp<=30].copy()
rv={}
for s in d.symbol.unique():
    bars=json.load(open(f"data/daily_raw/{s}.json"))["bars"]; px=pd.Series({x["date"]:x["close"] for x in bars}).sort_index()
    px=px[px.index<="2026-11-06"]; rv[s]=px
def fwd(r):
    p=rv[r.symbol]; seg=p[(p.index>=r.date)&(p.index<=r.expiry)]
    if len(seg)<4: return np.nan
    lr=np.diff(np.log(seg.values)); return float(np.sqrt(252*np.mean(lr**2)))
d["fwd_rv"]=d.apply(fwd,axis=1); d=d.dropna(subset=["fwd_rv"])
# one obs per contract-expiry-date (ATM-ish call/put share the date; average the pair)
k=d.groupby(["symbol","expiry","date"]).agg(obs_iv=("obs_iv","mean"),fwd_rv=("fwd_rv","first"),fc=("forecast_base_vol_tminus1","first")).reset_index()
k["iv_minus_rv"]=k.obs_iv-k.fwd_rv; k["var_ratio"]=k.obs_iv**2/k.fwd_rv**2
out=k.groupby("symbol").agg(n=("obs_iv","size"),n_expiries=("expiry","nunique"),med_iv=("obs_iv","median"),med_fwd_rv=("fwd_rv","median"),med_iv_minus_rv=("iv_minus_rv","median"),frac_iv_gt_rv=("iv_minus_rv",lambda x:(x>0).mean()),med_trailing_fc=("fc","median")).reset_index()
out.to_csv(f"{D}/{b}_realhist-vrp-outcome.csv",index=False); k.to_csv(f"{D}/{b}_realhist-vrp-rows.csv",index=False)
print(out.round(3).to_string())
