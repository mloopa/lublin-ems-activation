import sys, pandas as pd, numpy as np
o=pd.read_parquet('calls.parquet')
o=o.rename(columns={'team':'adult_','adult':'team'}).rename(columns={'adult_':'adult'})
c=pd.read_csv(sys.argv[1],encoding='utf-8-sig',usecols=[0,3,4,10],low_memory=False); c.columns=['county','t_dep','t_ret','zrm']
for x in ['t_dep','t_ret']: c[x]=pd.to_datetime(c[x],errors='coerce')
c=c.drop_duplicates(['zrm','t_ret'])
m=o.merge(c[['zrm','t_ret','county']],on=['zrm','t_ret'],how='left')
print(len(o),len(m),"matched",m.county.notna().mean().round(4))
# fallback: zrm -> modal county
mode=m.dropna(subset=['county']).groupby('zrm').county.agg(lambda s:s.value_counts().index[0])
share=m.dropna(subset=['county']).groupby('zrm').county.agg(lambda s:s.value_counts(normalize=True).iloc[0])
print("ZRM home-county purity median",share.median(), "min",share.min())
m['county_home']=m.zrm.map(mode)
print("home county available",m.county_home.notna().mean())
m.to_parquet('calls2.parquet'); print(m.team.value_counts().to_dict())
