import sys, pandas as pd, numpy as np
cols=['Numer zlecenia','Czas zlecenia ZRM','Czas wyjazdu ZRM','Czas powrotu ZRM','Identyfikator ZRM','Identyfikator pojazdu','Na sygnale/bez sygnału','Kod pilności','Typ zespołu','Dorosły/dziecko']
parts=[pd.read_excel(f,usecols=cols,dtype=str) for f in sys.argv[2:]]
df=pd.concat(parts,ignore_index=True)
df.columns=['id','t_order','t_dep','t_ret','zrm','veh','signal','prio','team','adult']
for c in ['t_order','t_dep','t_ret']: df[c]=pd.to_datetime(df[c],errors='coerce')
df.to_parquet(sys.argv[1])
print(len(df), "dup id", df.id.duplicated().sum())
print(df[['t_order','t_dep','t_ret']].isna().sum().to_dict(), df.t_order.min(), df.t_order.max())
rt=(df.t_dep-df.t_order).dt.total_seconds()
print("order->dep s", rt.quantile([.01,.05,.1,.25,.5,.75,.9,.95,.99]).round(1).to_dict())
print("neg",(rt<0).sum(),">86400",(rt>86400).sum(),"valid 0..86400",((rt>=0)&(rt<=86400)).sum())
print(df.signal.value_counts().to_dict()); print(df.team.value_counts().to_dict()); print(df.prio.value_counts().to_dict()); print(df.adult.value_counts().to_dict())
print("by year", df.t_order.dt.year.value_counts().sort_index().to_dict())
d=df[(rt>=0)&(rt<=86400)].copy(); d['rt']=rt
print("year median", d.groupby(d.t_order.dt.year).rt.median().to_dict())
print("hour median", d.groupby(d.t_order.dt.hour).rt.median().to_dict())
