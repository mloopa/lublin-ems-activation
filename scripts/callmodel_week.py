import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
d=pd.read_parquet('analytic.parquet'); day=pd.read_csv('out/daily_series.csv',index_col=0,parse_dates=True)
d['cases']=d.date.map(day.cases100k)
d['period']=np.where(d.phase.isin(['IPO','VRP']), d.phase+np.where(d.cases>=20,'_wave','_inter'), d.phase)
d['lrt']=np.log(d.rt+1); d['busy10']=d.busy_frac*10; d['week']=d.t_order.dt.strftime('%G-%V').factorize()[0]
f="lrt ~ C(period, Treatment('PPP')) + C(hour, Treatment(12)) + C(dow) + C(month) + urgent + spec + siren + busy10 + C(county_home)"
m=smf.ols(f,d).fit(cov_type='cluster',cov_kwds={'groups':d.week})
ci=m.conf_int(); keep=[k for k in m.params.index if any(s in k for s in ['period','hour','urgent','spec','siren','busy10'])]
t=pd.DataFrame({'pct':(np.exp(m.params[keep])-1)*100,'lo':(np.exp(ci.loc[keep,0])-1)*100,'hi':(np.exp(ci.loc[keep,1])-1)*100,'p':m.pvalues[keep]})
t.to_csv('out/callmodel_ols_week.csv'); print(t.round(1).to_string()); print('nweeks',d.week.nunique())
