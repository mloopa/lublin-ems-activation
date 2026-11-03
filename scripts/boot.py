import pandas as pd, numpy as np, json, time, warnings; warnings.filterwarnings('ignore')
import patsy, statsmodels.formula.api as smf
d=pd.read_parquet('analytic.parquet'); day=pd.read_csv('out/daily_series.csv',index_col=0,parse_dates=True)
d['cases']=d.date.map(day.cases100k)
d['period']=np.where(d.phase.isin(['IPO','VRP']), d.phase+np.where(d.cases>=20,'_wave','_inter'), d.phase)
d['lrt']=np.log(d.rt+1); d['busy10']=d.busy_frac*10
d['did']=d.date.factorize(sort=True)[0]; d['week']=d.t_order.dt.strftime('%G-%V')
f="lrt ~ C(period, Treatment('PPP')) + C(hour, Treatment(12)) + C(dow) + C(month) + urgent + spec + siren + busy10 + C(county_home)"
key=["C(period, Treatment('PPP'))[T.IPO_wave]","C(period, Treatment('PPP'))[T.IPO_inter]","C(period, Treatment('PPP'))[T.VRP_wave]","C(period, Treatment('PPP'))[T.VRP_inter]","C(period, Treatment('PPP'))[T.ETD]","C(hour, Treatment(12))[T.6]","C(hour, Treatment(12))[T.4]","urgent","spec","busy10"]
pct=lambda x:(np.exp(x)-1)*100
res={}
m0=smf.ols(f,d).fit()
for name,g in [('day','did'),('week','week'),('team','zrm')]:
    t=time.time(); m=m0.get_robustcov_results(cov_type='cluster',groups=d[g].factorize()[0]); 
    ci=m.conf_int(); idx=list(m0.params.index)
    res['cluster_'+name]={k:[pct(m.params[idx.index(k)]),pct(ci[idx.index(k),0]),pct(ci[idx.index(k),1])] for k in key}
    print(name,round(time.time()-t))
# moving block bootstrap over days (block 14 days, circular), 500 reps
y,X=patsy.dmatrices(f,d,return_type='dataframe'); cols=list(X.columns); Xv=X.values; yv=y.values.ravel()
D=d.did.values; nD=D.max()+1
XtX=np.zeros((nD,Xv.shape[1],Xv.shape[1])); Xty=np.zeros((nD,Xv.shape[1]))
order=np.argsort(D,kind='stable'); bounds=np.searchsorted(D[order],np.arange(nD+1))
for k in range(nD):
    ii=order[bounds[k]:bounds[k+1]]; Xi=Xv[ii]; XtX[k]=Xi.T@Xi; Xty[k]=Xi.T@yv[ii]
rng=np.random.default_rng(42); B=500; L=14; nb=int(np.ceil(nD/L)); kidx=[cols.index(k) for k in key]; draws=[]
for b in range(B):
    starts=rng.integers(0,nD,nb); days=((starts[:,None]+np.arange(L)[None,:])%nD).ravel()[:nD]
    w=np.bincount(days,minlength=nD).astype(float)
    A=np.tensordot(w,XtX,1); c=w@Xty
    draws.append(np.linalg.lstsq(A,c,rcond=None)[0][kidx])
draws=np.array(draws); bhat=m0.params[key].values
res['block_bootstrap_14d']={k:[pct(bhat[i]),pct(np.percentile(draws[:,i],2.5)),pct(np.percentile(draws[:,i],97.5))] for i,k in enumerate(key)}
# block length 28
draws2=[]
for b in range(300):
    L2=28; nb2=int(np.ceil(nD/L2)); starts=rng.integers(0,nD,nb2); days=((starts[:,None]+np.arange(L2)[None,:])%nD).ravel()[:nD]
    w=np.bincount(days,minlength=nD).astype(float); draws2.append(np.linalg.lstsq(np.tensordot(w,XtX,1),w@Xty,rcond=None)[0][kidx])
draws2=np.array(draws2)
res['block_bootstrap_28d']={k:[pct(bhat[i]),pct(np.percentile(draws2[:,i],2.5)),pct(np.percentile(draws2[:,i],97.5))] for i,k in enumerate(key)}
# seasonality sensitivity: Fourier instead of month; no seasonal adjustment
doy=d.t_order.dt.dayofyear; 
for kk in [1,2]: d[f's{kk}']=np.sin(2*np.pi*kk*doy/365.25); d[f'c{kk}']=np.cos(2*np.pi*kk*doy/365.25)
for nm,ff in [('fourier',f.replace('C(month)','s1+c1+s2+c2')),('no_season',f.replace(' + C(month)',''))]:
    m=smf.ols(ff,d).fit(cov_type='cluster',cov_kwds={'groups':d.did}); ci=m.conf_int()
    res['season_'+nm]={k:[pct(m.params[k]),pct(ci.loc[k,0]),pct(ci.loc[k,1])] for k in key[:5]}
json.dump(res,open('out/robustness.json','w'),indent=1,default=float)
for k,v in res.items():
    print('\n##',k)
    for t,(a,b,c) in v.items(): print(f'  {t[-22:]:24s} {a:+6.1f} [{b:+6.1f}, {c:+6.1f}]')
