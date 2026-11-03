import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
d=pd.read_parquet('analytic.parquet'); day=pd.read_csv('out/daily_series.csv',index_col=0,parse_dates=True)
d['cases']=d.date.map(day.cases100k)
d['period']=np.where(d.phase.isin(['IPO','VRP']), d.phase+np.where(d.cases>=20,'_wave','_inter'), d.phase)
d['lrt']=np.log(d.rt+1); d['busy10']=d.busy_frac*10; d['week']=d.t_order.dt.strftime('%G-%V').factorize()[0]
d['code']=np.where(d.urgent==1,'code1','code2'); d['limit_ok']=np.where(d.urgent==1,d.rt<=60,d.rt<=180)
R={}
def S(g): r=g.rt; return pd.Series({'orders':len(r),'median':r.median(),'q1':r.quantile(.25),'q3':r.quantile(.75),'p90':r.quantile(.9),'within_limit_pct':100*g.limit_ok.mean(),'occupancy_pct':100*g.busy_frac.mean()})
T1=d.groupby(['period','code']).apply(S); print(T1.round(1)); R['t_period_code']=T1
Tc=d.groupby('code').apply(S); print(Tc.round(1)); R['t_code']=Tc
Th=d.groupby(['hour','code']).rt.median().unstack(); R['t_hour_code']=Th
Tr=d.groupby(['region','code']).apply(S); print(Tr.round(1)); R['t_region_code']=Tr
Tcty=d.groupby(['county_home','code']).rt.median().unstack(); R['t_county_code']=Tcty
print('rank corr county code1 vs code2', Tcty.corr(method='spearman').iloc[0,1])
# priority share by period / merger
print('code1 share by period', d.groupby('period').urgent.mean().round(3).to_dict())
m=d.t_order>='2021-03-26'; print('code1 pre/post merger',d[~m].urgent.mean(),d[m].urgent.mean())
R['code1_share_merger']=pd.DataFrame({'pre_merger':[d[~m].urgent.mean()],'post_merger':[d[m].urgent.mean()],'orders_per_day_pre':[(~m).sum()/d[~m].date.nunique()],'orders_per_day_post':[m.sum()/d[m].date.nunique()]})
base="C(hour, Treatment(12)) + C(dow) + C(month) + spec + siren + busy10 + C(county_home)"
def fit(f,data,keys=None):
    mm=smf.ols(f,data).fit(cov_type='cluster',cov_kwds={'groups':data.week}); ci=mm.conf_int()
    ks=[k for k in mm.params.index if (keys is None and ('period' in k or k in('urgent','merged','busy10'))) or (keys and any(s in k for s in keys))]
    return pd.DataFrame({'pct':(np.exp(mm.params[ks])-1)*100,'lo':(np.exp(ci.loc[ks,0])-1)*100,'hi':(np.exp(ci.loc[ks,1])-1)*100,'p':mm.pvalues[ks]})
# stratified models
for c in ['code1','code2']:
    t=fit("lrt ~ C(period, Treatment('PPP')) + "+base, d[d.code==c],['period','busy10','hour, Treatment(12))[T.6]']); print(c); print(t.round(1)); R['model_'+c]=t
# interaction test
mi=smf.ols("lrt ~ C(period, Treatment('PPP'))*urgent + "+base, d).fit(cov_type='cluster',cov_kwds={'groups':d.week})
ints=[k for k in mi.params.index if ':urgent' in k]; W=mi.wald_test(' , '.join(f'{k} = 0' for k in ints),scalar=True)
print('interaction wald',W.statistic,W.pvalue); R['interaction_wald']=pd.DataFrame({'chi2':[float(W.statistic)],'df':[len(ints)],'p':[float(W.pvalue)]})
# IPO start 14 March
d2=d.copy(); br=pd.Timestamp('2020-03-14')
d2.loc[(d2.t_order<br),'period']='PPP'
t=fit("lrt ~ C(period, Treatment('PPP')) + urgent + "+base, d2); print('IPO 14 Mar'); print(t.round(1)); R['ipo_14march']=t
# exclusions sensitivity: negative intervals set to 0 and included
c=pd.read_parquet('calls2.parquet',columns=['id','t_order','t_dep']).drop_duplicates('id'); rt=(c.t_dep-c.t_order).dt.total_seconds()
neg=c[(rt<0)].id
x=pd.read_parquet('analytic_v9.parquet',columns=['id']) # analytic ids
print('neg n',len(neg))
R['excl_note']=pd.DataFrame({'negative_intervals':[len(neg)]})
pd.to_pickle(R,'out/rev1.pkl')
