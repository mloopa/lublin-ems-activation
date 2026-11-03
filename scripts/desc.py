import pandas as pd, numpy as np, json
d=pd.read_parquet('analytic.parquet')
def s(g):
    r=g.rt; return pd.Series({'n':len(r),'median':r.median(),'q1':r.quantile(.25),'q3':r.quantile(.75),'p90':r.quantile(.9),'pct_gt60':(r>60).mean()*100,'urgent':g.urgent.mean()*100,'busy':g.busy_frac.mean()*100,'mission':g.mission_min.median()})
out={}
for k in ['year','phase','hour','dow','month','urgent','spec','siren']:
    t=d.assign(_k=d[k]).groupby('_k').apply(s); out[k]=t; print('\n==',k); print(t.round(1).to_string())
ndays=d.groupby('phase').date.nunique(); print(ndays)
c=d.groupby('county_home').apply(s).sort_values('median'); out['county']=c; print(c.round(1).to_string())
pd.concat(out,names=['by','level']).to_csv('out/descriptives.csv')
print('overall',s(d).round(1).to_dict())
