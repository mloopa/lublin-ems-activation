import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
day=pd.read_csv('out/daily_series.csv',index_col=0,parse_dates=True); day['ln_n']=np.log(day.n)
dow=[c for c in day.columns if c.startswith('d_')]
base=['IPO','VRP','ETD','IPO_t','VRP_t','ETD_t','s1','c1','s2','c2','hol']+dow
adj=base+['lcases','ln_n','urg','busy','mission']
diag=[]; coefs=[]
for y,X,name in [('lmed',base,'daily median'),('lp90',base,'daily 90th percentile'),('lmed',adj,'daily median, adjusted')]:
    m=SARIMAX(day[y],exog=sm.add_constant(day[X].astype(float)),order=(1,0,1)).fit(disp=False,method='lbfgs',maxiter=3000)
    lb=acorr_ljungbox(m.resid,lags=[7,14,28],return_df=True).lb_pvalue
    diag.append(dict(model=name,arma='(1,0,1)',n_days=len(day),loglik=m.llf,AIC=m.aic,BIC=m.bic,ar1=m.params['ar.L1'],ma1=m.params['ma.L1'],sigma2=m.params['sigma2'],LB_p_lag7=lb.iloc[0],LB_p_lag14=lb.iloc[1],LB_p_lag28=lb.iloc[2]))
    ci=m.conf_int()
    for k in m.params.index: coefs.append(dict(model=name,term=k,estimate_log=m.params[k],ci_lo_log=ci.loc[k,0],ci_hi_log=ci.loc[k,1],p=m.pvalues[k],pct_change=(np.exp(m.params[k])-1)*100 if k not in('sigma2',) else np.nan))
its=json.load(open('out/its_final.json')); ends=[]
for mdl,r in its.items():
    for k,v in r.items():
        if isinstance(v,list) and len(v)==4: ends.append(dict(model=mdl,term=k,pct=v[0],ci_lo=v[1],ci_hi=v[2],p=v[3]))
X=sm.add_constant(day[base].astype(float)); hac=[]
for lag in [7,14,28]:
    o=sm.OLS(day.lmed,X).fit(cov_type='HAC',cov_kwds={'maxlags':lag})
    for k in ['IPO','IPO_t','VRP','VRP_t','ETD','ETD_t']:
        hac.append(dict(model=f'OLS Newey West lag {lag}',term=k,pct=(np.exp(o.params[k])-1)*100,ci_lo=(np.exp(o.conf_int().loc[k,0])-1)*100,ci_hi=(np.exp(o.conf_int().loc[k,1])-1)*100,p=o.pvalues[k]))
nb=sm.NegativeBinomial(day.n,X).fit(disp=False,cov_type='HAC',cov_kwds={'maxlags':14}); ci=nb.conf_int()
dem=pd.DataFrame({'term':nb.params.index,'IRR':np.exp(nb.params.values),'ci_lo':np.exp(ci[0].values),'ci_hi':np.exp(ci[1].values),'p':nb.pvalues.values})
R=pd.read_pickle('out/rev1.pkl'); E=pd.read_pickle('out/rev1_excl.pkl')
ex=json.load(open('out/exclusions.json')); rob=json.load(open('out/robustness.json')); mg=json.load(open('out/merger.json'))
d=pd.read_parquet('analytic.parquet'); d['ym']=d.t_order.dt.to_period('M').astype(str)
mon=d.groupby('ym').agg(orders=('rt','size'),median_s=('rt','median'),p90=('rt',lambda x:x.quantile(.9)),code1_pct=('urgent',lambda x:100*x.mean()),occupancy_pct=('busy_frac',lambda x:100*x.mean()),mission_min=('mission_min','median'))
mon['cases_per100k_day']=day.cases100k.groupby(day.index.to_period('M').astype(str)).mean()
cty=d.groupby(['region','county_home']).rt.agg(orders='size',median='median',p90=lambda x:x.quantile(.9)).reset_index()
cc=R['t_county_code']; cty['median_code1']=cty.county_home.map(cc['code1']); cty['median_code2']=cty.county_home.map(cc['code2'])
contents=pd.DataFrame([
 ('S1','Sample flow and excluded records by epidemic period and former dispatch area'),
 ('S2','Activation interval by year, phase, hour, weekday, month, priority, team type and lights and siren'),
 ('S3','Monthly series of activation interval, demand, priority 1 share, occupancy, mission duration and COVID-19 cases'),
 ('S4','Interrupted time series models: diagnostics (ARMA order, AIC, BIC, AR and MA coefficients, Ljung and Box tests), all coefficients, end of phase contrasts and least squares with Newey and West errors'),
 ('S5','Negative binomial model of daily dispatch orders (Newey and West standard errors)'),
 ('S6','Model for individual orders: all coefficients, standard errors clustered by calendar week'),
 ('S7','Robustness of confidence intervals: clustering by day, week and team, moving block bootstrap (14 and 28 days), seasonal adjustment'),
 ('S8','Median regression (effects in seconds), kernel sandwich confidence intervals'),
 ('S9','Sensitivity to the wave threshold (10, 20 and 30 cases per 100,000 per day)'),
 ('S10','Sensitivity to the start of the initial outbreak phase (14 March 2020)'),
 ('S11','Results by priority code: descriptive statistics, compliance with statutory limits, stratified models and interaction test'),
 ('S12','Merger of dispatch centres: time series and order models with a merger step, regional series, priority 1 share before and after'),
 ('S13','Station counties and former dispatch areas: medians, 90th percentiles and medians by priority code and year'),
 ('S14','Sensitivity to excluded records (negative intervals set to zero)')],columns=['table','content'])
f='out/Supplementary_Tables_S1-S14.xlsx'
with pd.ExcelWriter(f) as w:
    contents.to_excel(w,sheet_name='Contents',index=False)
    s1=pd.DataFrame([ex]).T.rename(columns={0:'n'})
    s1.to_excel(w,sheet_name='S1_flow_exclusions'); E['excl_period'].round(2).to_excel(w,sheet_name='S1_flow_exclusions',startrow=10); E['excl_region'].round(2).to_excel(w,sheet_name='S1_flow_exclusions',startrow=20)
    pd.read_csv('out/descriptives.csv').round(2).to_excel(w,sheet_name='S2_strata',index=False)
    mon.round(2).to_excel(w,sheet_name='S3_monthly')
    pd.DataFrame(diag).round(4).to_excel(w,sheet_name='S4_ITS',index=False)
    pd.DataFrame(ends+hac).round(3).to_excel(w,sheet_name='S4_ITS',startrow=6,index=False)
    pd.DataFrame(coefs).round(4).to_excel(w,sheet_name='S4_ITS_all_coefficients',index=False)
    dem.round(4).to_excel(w,sheet_name='S5_demand',index=False)
    pd.read_csv('out/callmodel_ols_week.csv').round(3).to_excel(w,sheet_name='S6_order_model',index=False)
    pd.DataFrame([dict(method=a,term=k,pct=v[0],ci_lo=v[1],ci_hi=v[2]) for a,b in rob.items() for k,v in b.items()]).round(2).to_excel(w,sheet_name='S7_robustness',index=False)
    q=pd.read_csv('out/callmodel_qr50.csv'); q['ci_method']='kernel sandwich, Hall and Sheather bandwidth, assumes independence'; q.round(3).to_excel(w,sheet_name='S8_median_regression',index=False)
    pd.read_csv('out/sens_threshold.csv').round(2).to_excel(w,sheet_name='S9_wave_threshold',index=False)
    R['ipo_14march'].round(3).to_excel(w,sheet_name='S10_IPO_14_March')
    R['t_period_code'].round(2).to_excel(w,sheet_name='S11_priority'); R['model_code1'].round(3).to_excel(w,sheet_name='S11_priority',startrow=16); R['model_code2'].round(3).to_excel(w,sheet_name='S11_priority',startrow=28); R['interaction_wald'].to_excel(w,sheet_name='S11_priority',startrow=40); R['t_hour_code'].round(1).to_excel(w,sheet_name='S11_priority',startrow=45)
    pd.DataFrame([dict(series_spec=a,term=k,pct=v[0],ci_lo=v[1],ci_hi=v[2]) for a,b in mg['its'].items() for k,v in b.items()]+[dict(series_spec='order model with merger step',term=k,pct=v[0],ci_lo=v[1],ci_hi=v[2]) for k,v in mg['order_merged'].items()]).round(2).to_excel(w,sheet_name='S12_merger',index=False)
    R['code1_share_merger'].round(3).to_excel(w,sheet_name='S12_merger',startrow=40,index=False)
    cty.round(1).to_excel(w,sheet_name='S13_counties_areas',index=False); R['t_region_code'].round(1).to_excel(w,sheet_name='S13_counties_areas',startrow=28)
    pd.DataFrame(mg['region_period']).to_excel(w,sheet_name='S13_counties_areas',startrow=40); pd.DataFrame(mg['region_year']).T.to_excel(w,sheet_name='S13_counties_areas',startrow=47)
    E['excl_sens'].to_excel(w,sheet_name='S14_exclusion_sensitivity')
print(pd.DataFrame(diag).round(3).to_string())
