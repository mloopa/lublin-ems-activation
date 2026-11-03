import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings('ignore')
import logging
import statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
d=pd.read_parquet('analytic.parquet')
day=d.groupby('date').agg(n=('rt','size'),med=('rt','median'),p90=('rt',lambda x:x.quantile(.9)),urg=('urgent','mean'),busy=('busy_frac','mean'),teams=('teams_day','first'),mission=('mission_min','median'))
idx=pd.date_range('2020-01-01','2022-12-31'); day=day.reindex(idx); print('missing days',day.n.isna().sum())
cs=pd.read_csv('pl_cases.csv',parse_dates=['date']).set_index('date').Poland.reindex(idx).fillna(0)
day['cases']=cs.rolling(7,min_periods=1).mean()/38.0*1e-0  # per 1M ->/38 => per 100k? 38M pop: cases/380 per 100k
day['cases100k']=cs.rolling(7,min_periods=1).mean()/380.0
day['lcases']=np.log1p(day.cases100k)
t=np.arange(len(idx)); day['t']=t/365.25
br={'IPO':'2020-03-04','VRP':'2021-01-15','ETD':'2022-05-16'}
ends={'IPO':'2021-01-15','VRP':'2022-05-16','ETD':'2023-01-01'}
for k,v in br.items():
    inph=(idx>=v)&(idx<ends[k])
    day[k]=inph.astype(int); day[k+'_t']=np.where(inph,(idx-pd.Timestamp(v)).days/365.25,0)
for k in [1,2]:
    day[f's{k}']=np.sin(2*np.pi*k*idx.dayofyear/365.25); day[f'c{k}']=np.cos(2*np.pi*k*idx.dayofyear/365.25)
dow=pd.get_dummies(idx.dayofweek,prefix='d',drop_first=True).astype(int); dow.index=idx; day=day.join(dow)
# Polish public holidays 2020-2022 (fixed + Easter Monday, Corpus Christi)
hol=['01-01','01-06','05-01','05-03','08-15','11-01','11-11','12-25','12-26']
mov=['2020-04-12','2020-04-13','2020-06-11','2021-04-04','2021-04-05','2021-06-03','2022-04-17','2022-04-18','2022-06-16']
day['hol']=[int(x.strftime('%m-%d') in hol or x.strftime('%Y-%m-%d') in mov) for x in idx]
day['lmed']=np.log(day.med); day['lp90']=np.log(day.p90)
day.to_csv('out/daily_series.csv')
base=[f'{k}' for k in br]+[f'{k}_t' for k in br]+['s1','c1','s2','c2','hol']+list(dow.columns)
res={}
def sarimax(y,X,name):
    best=None
    for order in [(1,0,0),(1,0,1),(2,0,0),(2,0,1)]:
        for so in [(0,0,0,0),(1,0,0,7)]:
            m=SARIMAX(day[y],exog=sm.add_constant(day[X]),order=order,seasonal_order=so).fit(disp=False)
            if best is None or m.aic<best.aic: best=m; bo=(order,so)
    lb=acorr_ljungbox(best.resid,lags=[7,14,28],return_df=True).lb_pvalue.round(3).tolist()
    ci=best.conf_int()
    r={'order':str(bo),'aic':round(best.aic,1),'ljungbox_p':lb,'coef':{c:[best.params[c],ci.loc[c,0],ci.loc[c,1],best.pvalues[c]] for c in X}}
    res[name]=r; print('\n##',name,bo,'AIC',round(best.aic,1),'LB p',lb)
    for c in X:
        if c in ['t','IPO','VRP','ETD','IPO_t','VRP_t','ETD_t','lcases','busy','urg','ln_n','mission']:
            p=best.params[c]; print(f"  {c:7s} {p:+.4f} [{ci.loc[c,0]:+.4f},{ci.loc[c,1]:+.4f}] p={best.pvalues[c]:.4f}  ratio={np.exp(p):.3f}")
    return best
day['ln_n']=np.log(day.n)
# 1) RT phases ITS
sarimax('lmed',base,'rt_median_phases')
sarimax('lp90',base,'rt_p90_phases')
# 2) RT with epidemiological + operational covariates (waves as continuous)
cov=['lcases','ln_n','urg','busy','mission','s1','c1','s2','c2','hol']+list(dow.columns)
sarimax('lmed',cov,'rt_median_covariates')
sarimax('lmed',base+['lcases','ln_n','urg','busy','mission'],'rt_median_phases_plus_cov')
# 3) Demand: NB GLM with HAC + SARIMAX on log counts
X=sm.add_constant(day[base].astype(float))
nb=sm.GLM(day.n,X,family=sm.families.NegativeBinomial(alpha=0.01)).fit(cov_type='HAC',cov_kwds={'maxlags':14})
# estimate alpha via ML
nbm=sm.NegativeBinomial(day.n,X).fit(disp=False,cov_type='HAC',cov_kwds={'maxlags':14})
print('\n## demand NB (HAC 14)  alpha',round(nbm.params['alpha'],4))
dem={}
for c in ['IPO','VRP','ETD','IPO_t','VRP_t','ETD_t']:
    p=nbm.params[c]; ci=nbm.conf_int().loc[c]; dem[c]=[p,ci[0],ci[1],nbm.pvalues[c]]
    print(f"  {c:7s} IRR={np.exp(p):.3f} [{np.exp(ci[0]):.3f},{np.exp(ci[1]):.3f}] p={nbm.pvalues[c]:.4f}")
pr=nbm.resid_pearson if hasattr(nbm,'resid_pearson') else (day.n-nbm.predict())/np.sqrt(nbm.predict())
print('  LB on Pearson resid',acorr_ljungbox((day.n-nbm.predict())/np.sqrt(nbm.predict()),lags=[7,14],return_df=True).lb_pvalue.round(4).tolist())
res['demand_nb_hac']={'alpha':nbm.params['alpha'],'coef':dem}
sarimax('ln_n',base,'demand_logcount_sarimax')
# naive KW vs model: for reviewer comparison
json.dump(res,open('out/its_results.json','w'),indent=1,default=float)
