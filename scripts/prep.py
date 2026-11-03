import pandas as pd, numpy as np, json
m=pd.read_parquet('calls2.parquet')
N0=len(m); m=m.drop_duplicates('id'); N1=len(m)
m['rt']=(m.t_dep-m.t_order).dt.total_seconds()
ex={'records':N0,'dup':N0-N1,'no_dep':int(m.t_dep.isna().sum()),'neg':int((m.rt<0).sum()),'gt24h':int((m.rt>86400).sum())}
d=m[(m.rt>=0)&(m.rt<=86400)].copy(); ex['analytic']=len(d)
# mission end for occupancy
d['end']=d.t_ret.where((d.t_ret>d.t_order)&((d.t_ret-d.t_order)<pd.Timedelta(hours=12)), d.t_order+pd.Timedelta(minutes=60))
# busy at order time, region and home county
def busy(df):
    s=np.sort(df.t_order.values); e=np.sort(df.end.values); t=df.t_order.values
    return np.searchsorted(s,t,side='right')-np.searchsorted(e,t,side='right')-1
d=d.sort_values('t_order'); d['busy_reg']=busy(d)
d['busy_cty']=d.groupby('county_home',group_keys=False).apply(lambda g: pd.Series(busy(g),index=g.index))
d['date']=d.t_order.dt.normalize()
teams=d.groupby('date').zrm.nunique(); d['teams_day']=d.date.map(teams)
tc=d.groupby(['county_home','date']).zrm.nunique(); d['teams_cty']=pd.MultiIndex.from_frame(d[['county_home','date']]).map(tc)
d['busy_frac']=d.busy_reg/d.teams_day
d['busy_frac_cty']=(d.busy_cty/d.teams_cty).clip(0,1.5)
d['mission_min']=(d.end-d.t_order).dt.total_seconds()/60
br=[pd.Timestamp(x) for x in ['2020-03-04','2021-01-15','2022-05-16']]
d['phase']=np.select([d.t_order<br[0],d.t_order<br[1],d.t_order<br[2]],['PPP','IPO','VRP'],'ETD')
d['hour']=d.t_order.dt.hour; d['dow']=d.t_order.dt.dayofweek; d['month']=d.t_order.dt.month; d['year']=d.t_order.dt.year
d['urgent']=(d.prio.astype(str)=='1').astype(int); d['spec']=(d.team=='S').astype(int); d['siren']=(d.signal=='NA SYGNALE').astype(int)
d.drop(columns=['veh']).to_parquet('analytic.parquet')
json.dump(ex,open('out/exclusions.json','w'),indent=1); print(ex)
print(d[['busy_reg','teams_day','busy_frac','busy_frac_cty','mission_min']].describe().round(3))
