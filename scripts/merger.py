import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.tsa.statespace.sarimax import SARIMAX
d=pd.read_parquet('analytic.parquet'); day=pd.read_csv('out/daily_series.csv',index_col=0,parse_dates=True)
out={}
# medians by region and MZ periods / year
per=pd.cut(d.t_order,[pd.Timestamp('2019-12-31'),pd.Timestamp('2021-03-01'),pd.Timestamp('2021-03-26'),pd.Timestamp('2021-05-01'),pd.Timestamp('2023-01-01')],labels=['01.2020-02.2021','1-25.03.2021','26.03-30.04.2021','05.2021-12.2022'],right=False)
t1=d.groupby(['region',per]).rt.median().unstack(); t2=d.groupby(['region','year']).rt.agg(['median',lambda x:x.quantile(.9)]).unstack()
print(t1); print(d.groupby(['region','year']).rt.median().unstack()); print(d.groupby('region').rt.agg(['size','median']))
out['region_period']=t1.to_dict(); out['region_year']=d.groupby(['region','year']).rt.median().unstack().to_dict()
# ITS with merger step on regional daily medians and on total
idx=day.index; day['MRG']=(idx>='2021-03-26').astype(int); day['MRG_t']=np.where(idx>='2021-03-26',(idx-pd.Timestamp('2021-03-26')).days/365.25,0)
dow=[c for c in day.columns if c.startswith('d_')]
base=['IPO','VRP','ETD','IPO_t','VRP_t','ETD_t','s1','c1','s2','c2','hol']+dow
yrs=lambda a,b:(pd.Timestamp(b)-pd.Timestamp(a)).days/365.25
def fit(y,X):
    best=None
    for o in [(1,0,1),(1,0,0),(2,0,1)]:
        m=SARIMAX(y,exog=sm.add_constant(X.astype(float)),order=o).fit(disp=False,method='lbfgs',maxiter=3000)
        if best is None or m.aic<best.aic: best=m
    return best
def lc(m,w):
    P=m.params; V=m.cov_params(); L=np.zeros(len(P)); n=list(P.index)
    for k,v in w.items(): L[n.index(k)]=v
    e=L@P.values; s=np.sqrt(L@V.values@L); return [(np.exp(e)-1)*100,(np.exp(e-1.96*s)-1)*100,(np.exp(e+1.96*s)-1)*100]
res={}
for spec,extra in [('phases',[]),('phases+merger_level',['MRG']),('phases+merger_level_slope',['MRG','MRG_t'])]:
    m=fit(day.lmed,day[base+extra]); r={'VRP_t':lc(m,{'VRP_t':1}),'VRP end':lc(m,{'VRP':1,'VRP_t':yrs('2021-01-15','2022-05-15')}),'ETD end':lc(m,{'ETD':1,'ETD_t':yrs('2022-05-16','2022-12-31')}),'IPO end':lc(m,{'IPO':1,'IPO_t':yrs('2020-03-04','2021-01-14')})}
    if 'MRG' in extra: r['MRG level']=lc(m,{'MRG':1})
    if 'MRG_t' in extra: r['MRG slope']=lc(m,{'MRG_t':1})
    res['total|'+spec]=r
for g in ['Lublin','Biała Podlaska','Chełm','Zamość']:
    s=d[d.region==g].groupby('date').rt.median().reindex(idx); y=np.log(s.interpolate())
    for spec,extra in [('phases',[]),('phases+merger_level',['MRG'])]:
        m=fit(y,day[base+extra]); r={'VRP_t':lc(m,{'VRP_t':1}),'IPO end':lc(m,{'IPO':1,'IPO_t':yrs('2020-03-04','2021-01-14')}),'ETD end':lc(m,{'ETD':1,'ETD_t':yrs('2022-05-16','2022-12-31')})}
        if extra: r['MRG level']=lc(m,{'MRG':1})
        res[g+'|'+spec]=r
for k,v in res.items():
    print(k); [print(f'   {a:10s} {b[0]:+.1f} ({b[1]:+.1f} to {b[2]:+.1f})') for a,b in v.items()]
out['its']=res
# order-level with merger dummy
day2=day; d['cases']=d.date.map(day.cases100k)
d['period']=np.where(d.phase.isin(['IPO','VRP']), d.phase+np.where(d.cases>=20,'_wave','_inter'), d.phase)
d['lrt']=np.log(d.rt+1); d['busy10']=d.busy_frac*10; d['week']=d.t_order.dt.strftime('%G-%V').factorize()[0]; d['merged']=(d.t_order>='2021-03-26').astype(int)
f="lrt ~ C(period, Treatment('PPP')) + merged + C(hour, Treatment(12)) + C(dow) + C(month) + urgent + spec + siren + busy10 + C(county_home)"
mm=smf.ols(f,d).fit(cov_type='cluster',cov_kwds={'groups':d.week}); ci=mm.conf_int()
o={k:[(np.exp(mm.params[k])-1)*100,(np.exp(ci.loc[k,0])-1)*100,(np.exp(ci.loc[k,1])-1)*100] for k in mm.params.index if 'period' in k or k=='merged'}
for k,v in o.items(): print('ORD',k[-12:],[round(x,1) for x in v])
out['order_merged']=o
json.dump(out,open('out/merger.json','w'),indent=1,default=str)
