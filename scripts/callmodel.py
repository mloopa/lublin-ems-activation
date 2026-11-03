import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
d=pd.read_parquet('analytic.parquet'); day=pd.read_csv('out/daily_series.csv',index_col=0,parse_dates=True)
d['cases']=d.date.map(day.cases100k)
d['period']=np.where(d.phase.isin(['IPO','VRP']), d.phase+np.where(d.cases>=20,'_wave','_inter'), d.phase)
d['lrt']=np.log(d.rt+1); d['busy10']=d.busy_frac*10; d['busyc10']=d.busy_frac_cty*10
d['hband']=pd.cut(d.hour,[-1,5,6,7,17,21,23],labels=['00-05','06','07','08-17','18-21','22-23'])
d['did']=d.date.factorize()[0]
f="lrt ~ C(period, Treatment('PPP')) + C(hour, Treatment(12)) + C(dow) + C(month) + urgent + spec + siren + busy10 + C(county_home)"
m=smf.ols(f,d).fit(cov_type='cluster',cov_kwds={'groups':d.did})
ci=m.conf_int(); keep=[k for k in m.params.index if any(s in k for s in ['period','hour','urgent','spec','siren','busy10'])]
tab=pd.DataFrame({'pct':(np.exp(m.params[keep])-1)*100,'lo':(np.exp(ci.loc[keep,0])-1)*100,'hi':(np.exp(ci.loc[keep,1])-1)*100,'p':m.pvalues[keep]})
print('N',int(m.nobs),'R2',round(m.rsquared,3)); print(tab.round(2).to_string())
tab.to_csv('out/callmodel_ols.csv')
# sensitivity: county-level occupancy instead of regional
m2=smf.ols(f.replace('busy10','busyc10'),d).fit(cov_type='cluster',cov_kwds={'groups':d.did})
print('county busy10', (np.exp(m2.params['busyc10'])-1)*100, (np.exp(m2.conf_int().loc['busyc10'])-1)*100)
# median regression on 150k random sample
s=d.sample(150000,random_state=1)
q=smf.quantreg(f.replace('lrt','rt'),s).fit(q=0.5,max_iter=5000)
kq=[k for k in keep if k in q.params.index]
qt=pd.DataFrame({'sec':q.params[kq],'lo':q.conf_int().loc[kq,0],'hi':q.conf_int().loc[kq,1],'p':q.pvalues[kq]})
print(qt.round(2).to_string()); qt.to_csv('out/callmodel_qr50.csv')
json.dump({'n':int(m.nobs),'r2':m.rsquared,'n_qr':150000},open('out/callmodel_meta.json','w'))
