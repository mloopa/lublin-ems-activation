import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
import statsmodels, statsmodels.formula.api as smf
from scipy.stats import spearmanr
print('statsmodels',statsmodels.__version__)
d=pd.read_parquet('analytic.parquet'); day=pd.read_csv('out/daily_series.csv',index_col=0,parse_dates=True)
d['cases']=d.date.map(day.cases100k); d['lrt']=np.log(d.rt+1); d['busy10']=d.busy_frac*10; d['did']=d.date.factorize()[0]
rows=[]
for thr in [10,20,30]:
    d['period']=np.where(d.phase.isin(['IPO','VRP']), d.phase+np.where(d.cases>=thr,'_wave','_inter'), d.phase)
    m=smf.ols("lrt ~ C(period, Treatment('PPP')) + C(hour) + C(dow) + C(month) + urgent + spec + siren + busy10 + C(county_home)",d).fit(cov_type='cluster',cov_kwds={'groups':d.did})
    for k in m.params.index:
        if 'period' in k:
            ci=m.conf_int().loc[k]; rows.append(dict(threshold=thr,term=k.split('T.')[1].rstrip(']'),pct=(np.exp(m.params[k])-1)*100,lo=(np.exp(ci[0])-1)*100,hi=(np.exp(ci[1])-1)*100))
s=pd.DataFrame(rows); print(s.round(1).to_string()); s.to_csv('out/sens_threshold.csv',index=False)
a=d[d.phase=='PPP'].groupby('county_home').rt.median(); b=d[d.phase!='PPP'].groupby('county_home').rt.median()
r=spearmanr(a,b); print('county rank PPP vs pandemic rho',r)
pd.DataFrame({'PPP_median':a,'pandemic_median':b}).to_csv('out/county_rank.csv')
