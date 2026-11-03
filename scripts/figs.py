import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.dates as mdates
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#52514e','axes.labelcolor':'#0b0b0b','xtick.color':'#52514e','ytick.color':'#52514e','axes.grid':True,'grid.color':'#e6e5e0','grid.linewidth':0.6,'savefig.dpi':300})
B,O,A,G='#2a78d6','#eb6834','#1baf7a','#8a8984'
day=pd.read_csv('out/daily_series.csv',index_col=0,parse_dates=True)
d=pd.read_parquet('analytic.parquet')
ph=[('PPP','2020-01-01','2020-03-03'),('IPO','2020-03-04','2021-01-14'),('VRP','2021-01-15','2022-05-15'),('ETD','2022-05-16','2022-12-31')]
def shade(ax,lab=False):
    for i,(n,a,b) in enumerate(ph):
        if i%2: ax.axvspan(pd.Timestamp(a),pd.Timestamp(b)+pd.Timedelta(days=1),color='#f1f0ec',zorder=0,lw=0)
        if lab: ax.text(pd.Timestamp(a)+(pd.Timestamp(b)-pd.Timestamp(a))/2,0.96,n,transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=8,fontweight='bold',color='#52514e')
# Fig 2: four stacked panels
fig,ax=plt.subplots(4,1,figsize=(7.2,8.2),sharex=True)
for a in ax: shade(a)
shade(ax[0],True)
ax[0].plot(day.index,day.med,color=B,lw=0.5,alpha=0.35); ax[0].plot(day.index,day.med.rolling(14,center=True).mean(),color=B,lw=2)
ax[0].set_ylabel('Median activation\ninterval (s)'); ax[0].set_ylim(30,92); ax[0].set_title('A  Daily median activation interval (thin) and two week mean (thick)',loc='left',fontsize=9)
ax[1].plot(day.index,day.p90.rolling(14,center=True).mean(),color=O,lw=2); ax[1].set_ylabel('90th percentile (s)'); ax[1].set_title('B  90th percentile, two week mean',loc='left',fontsize=9)
ax[2].plot(day.index,day.n,color=G,lw=0.5,alpha=0.5); ax[2].plot(day.index,day.n.rolling(14,center=True).mean(),color='#0b0b0b',lw=1.6); ax[2].set_ylabel('Dispatch orders\nper day'); ax[2].set_title('C  Daily dispatch orders',loc='left',fontsize=9)
ax[3].fill_between(day.index,0,day.cases100k,color=A,alpha=0.35,lw=0); ax[3].plot(day.index,day.cases100k,color=A,lw=1); ax[3].axhline(20,color='#52514e',ls=(0,(3,3)),lw=0.8); ax[3].text(pd.Timestamp('2020-01-10'),22,'wave threshold, 20 per 100k',fontsize=7,color='#52514e')
ax[3].set_ylabel('COVID-19 cases\nper 100k per day'); ax[3].set_title('D  Notified COVID-19 cases in Poland',loc='left',fontsize=9)
ax[3].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10])); ax[3].xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
fig.tight_layout(); fig.savefig('fig/Fig2_timeseries.png'); fig.savefig('fig/Fig2_timeseries.tiff',pil_kwargs={'compression':'tiff_lzw'})
# Fig 3: hourly
h=d.groupby('hour').agg(med=('rt','median'),q1=('rt',lambda x:x.quantile(.25)),q3=('rt',lambda x:x.quantile(.75)),busy=('busy_frac','mean'),n=('rt','size'))
ndays=d.date.nunique(); h['perh']=h.n/ndays
fig,ax=plt.subplots(3,1,figsize=(6.4,6.4),sharex=True)
hc=d.groupby(['hour','urgent']).rt.median().unstack()
ax[0].plot(hc.index,hc[1],color=O,lw=2,marker='o',ms=4,label='Priority code 1'); ax[0].plot(hc.index,hc[0],color=B,lw=2,marker='s',ms=4,label='Priority code 2'); ax[0].set_ylabel('Median activation\ninterval (s)'); ax[0].set_title('A  Median activation interval by priority code',loc='left',fontsize=9); ax[0].legend(frameon=False,fontsize=8,loc='upper right')
ax[1].bar(h.index,h.perh,color=G,width=0.8); ax[1].set_ylabel('Orders per hour'); ax[1].set_title('B  Mean dispatch orders per clock hour',loc='left',fontsize=9)
ax[2].plot(h.index,h.busy*100,color='#0b0b0b',lw=2,marker='o',ms=4); ax[2].set_ylabel('Teams on a mission (%)'); ax[2].set_title('C  Fleet occupancy at the moment of the order',loc='left',fontsize=9)
ax[2].set_xticks(range(0,24,2)); ax[2].set_xlabel('Hour of day (order time)')
fig.tight_layout(); fig.savefig('fig/Fig3_hourly.png'); fig.savefig('fig/Fig3_hourly.tiff',pil_kwargs={'compression':'tiff_lzw'})
# Fig 4: forest plot
t=pd.read_csv('out/callmodel_ols_week.csv',index_col=0)
lab={}
for k in t.index:
    if 'period' in k: lab[k]='Period: '+k.split('T.')[1].rstrip(']').replace('_inter',' between waves').replace('_wave',' wave days')
    elif 'hour' in k: lab[k]='Hour '+k.split('T.')[1].rstrip(']').zfill(2)+':00'
lab.update({'urgent':'Priority code 1 (vs 2)','spec':'Specialist team S (vs basic team P)','siren':'Lights and siren used','busy10':'Fleet occupancy (+10 pp)'})
order=[k for k in t.index if 'period' in k]+[k for k in t.index if 'hour' in k and k.split('T.')[1].rstrip(']') in ['0','2','4','6','7','8','18','23']]+['urgent','spec','siren','busy10']
tt=t.loc[order]; y=np.arange(len(tt))[::-1]
fig,ax=plt.subplots(figsize=(6.4,5.6))
ax.axvline(0,color='#52514e',lw=0.8)
ax.errorbar(tt.pct,y,xerr=[tt.pct-tt.lo,tt.hi-tt.pct],fmt='o',color=B,ms=5,elinewidth=1.4,capsize=0)
ax.set_yticks(y); ax.set_yticklabels([lab[k] for k in order]); ax.set_xlabel('Adjusted difference in activation interval (%, 95% CI)')
ax.grid(axis='y',visible=False)
fig.tight_layout(); fig.savefig('fig/Fig4_forest.png'); fig.savefig('fig/Fig4_forest.tiff',pil_kwargs={'compression':'tiff_lzw'})
# Fig 5: counties, coloured by former dispatch region
c=d.groupby('county_home').agg(med=('rt','median'),q1=('rt',lambda x:x.quantile(.25)),q3=('rt',lambda x:x.quantile(.75)),p90=('rt',lambda x:x.quantile(.9)),region=('region','first')).sort_values('med')
names=[s.replace('powiat ','') for s in c.index]; y=np.arange(len(c))
RC={'Lublin':B,'Biała Podlaska':A,'Chełm':O,'Zamość':'#6250d6'}
fig,ax=plt.subplots(figsize=(6.4,6.4))
for i,(k,r) in enumerate(c.iterrows()):
    col=RC[r.region]; ax.hlines(i,r.q1,r.q3,color=col,lw=3,alpha=0.35); ax.plot(r.med,i,'o',color=col,ms=6); ax.plot(r.p90,i,'D',color=col,ms=4,mfc='white',mew=1.2)
ax.set_yticks(y); ax.set_yticklabels(names); ax.set_xlabel('Activation interval (s), by station county of the team'); ax.grid(axis='y',visible=False)
from matplotlib.lines import Line2D
h=[Line2D([0],[0],marker='o',color=RC[g],lw=0,ms=6,label=g) for g in ['Lublin','Biała Podlaska','Chełm','Zamość']]
h+= [Line2D([0],[0],marker='o',color='#52514e',lw=0,ms=6,label='Median (bar: IQR)'),Line2D([0],[0],marker='D',color='#52514e',mfc='white',lw=0,ms=5,label='90th percentile')]
ax.legend(handles=h,frameon=False,loc='lower right',title='Former dispatch region',title_fontsize=8,fontsize=8)
fig.tight_layout(); fig.savefig('fig/Fig5_counties.png'); fig.savefig('fig/Fig5_counties.tiff',pil_kwargs={'compression':'tiff_lzw'})
c.to_csv('out/county_rt.csv')
