import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
day=pd.read_csv('out/daily_series.csv',index_col=0,parse_dates=True); day['ln_n']=np.log(day.n)
dow=[c for c in day.columns if c.startswith('d_')]
base=['IPO','VRP','ETD','IPO_t','VRP_t','ETD_t','s1','c1','s2','c2','hol']+dow
adj=base+['lcases','ln_n','urg','busy','mission']
yrs=lambda a,b:(pd.Timestamp(b)-pd.Timestamp(a)).days/365.25
ends={'IPO end':{'IPO':1,'IPO_t':yrs('2020-03-04','2021-01-14')},'VRP end':{'VRP':1,'VRP_t':yrs('2021-01-15','2022-05-15')},'ETD end':{'ETD':1,'ETD_t':yrs('2022-05-16','2022-12-31')}}
out={}
for y,X,name in [('lmed',base,'median'),('lp90',base,'p90'),('lmed',adj,'median_adj')]:
    best=None
    for order in [(1,0,0),(1,0,1),(2,0,0),(2,0,1)]:
        for so in [(0,0,0,0),(1,0,0,7)]:
            m=SARIMAX(day[y],exog=sm.add_constant(day[X].astype(float)),order=order,seasonal_order=so).fit(disp=False,method='lbfgs',maxiter=2000)
            if best is None or m.aic<best.aic: best=m; bo=(order,so)
    m=best; P=m.params; V=m.cov_params(); names=list(P.index); r={'order':str(bo),'lb':acorr_ljungbox(m.resid,lags=[7,14,28],return_df=True).lb_pvalue.round(2).tolist()}
    def lc(w,scale=1):
        L=np.zeros(len(P)); 
        for k,v in w.items(): L[names.index(k)]=v*scale
        e=L@P.values; s=np.sqrt(L@V.values@L); return [(np.exp(e)-1)*100,(np.exp(e-1.96*s)-1)*100,(np.exp(e+1.96*s)-1)*100, float(2*(1-__import__('scipy').stats.norm.cdf(abs(e/s))))]
    for k in ['IPO','IPO_t','VRP','VRP_t','ETD','ETD_t']: r[k]=lc({k:1})
    for k,w in ends.items(): r[k]=lc(w)
    if name=='median_adj':
        r['lcases']=lc({'lcases':1}); r['busy10']=lc({'busy':0.1}); r['urg10']=lc({'urg':0.1})
    out[name]=r
    print('##',name,bo,r['lb'])
    for k,v in r.items():
        if isinstance(v,list) and len(v)==4: print(f'  {k:8s} {v[0]:+.1f} ({v[1]:+.1f} to {v[2]:+.1f}) p={v[3]:.3g}')
json.dump(out,open('out/its_final.json','w'),indent=1)
