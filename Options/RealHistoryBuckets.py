import pandas as pd, sys
D=sys.argv[1]; b="2026-11-09_options-validation"
d=pd.read_csv(f"{D}/{b}_realhist-iv.csv")
d["dte_bucket"]=pd.cut(d.days_to_exp,[0,7,14,30,60,400],labels=["1-7","8-14","15-30","31-60","61+"])
d["mny_bucket"]=pd.cut(d.moneyness,[0,0.95,0.99,1.01,1.05,9],labels=["<.95",".95-.99","ATM","1.01-1.05",">1.05"])
g=d.groupby(["symbol","dte_bucket"],observed=True).agg(n=("obs_iv","size"),med_ratio=("iv_over_forecast","median"),p25=("iv_over_forecast",lambda x:x.quantile(.25)),p75=("iv_over_forecast",lambda x:x.quantile(.75)),medAPE_m10=("err_pct_m1.0",lambda x:x.abs().median()),medAPE_m12=("err_pct_m1.2",lambda x:x.abs().median()),bias_m12=("err_pct_m1.2","median")).reset_index()
g.to_csv(f"{D}/{b}_realhist-by-dte.csv",index=False)
s=d[d.days_to_exp<=14].groupby("symbol").agg(n=("obs_iv","size"),med_ratio=("iv_over_forecast","median"),p25=("iv_over_forecast",lambda x:x.quantile(.25)),p75=("iv_over_forecast",lambda x:x.quantile(.75)),bias_m12=("err_pct_m1.2","median"),medAPE_m10=("err_pct_m1.0",lambda x:x.abs().median()))
print(s.round(3)); print(d.groupby("mny_bucket",observed=True).iv_over_forecast.median().round(3))
