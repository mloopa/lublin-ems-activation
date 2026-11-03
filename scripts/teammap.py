import pandas as pd, numpy as np
d=pd.read_parquet('calls2.parquet',columns=['zrm','veh','t_order','county','county_home'])
z=d.groupby('zrm').agg(first=('t_order','min'),last=('t_order','max'),n=('t_order','size'))
old=z[z['last']<'2021-03-27'].index; new=z[z['first']>'2021-03-20'].index
w0=d[(d.t_order>='2021-02-15')&(d.t_order<'2021-03-26')&d.zrm.isin(old)]
w1=d[(d.t_order>='2021-03-23')&(d.t_order<'2021-05-01')&d.zrm.isin(new)]
vo=w0.groupby('zrm').veh.agg(lambda s:s.value_counts().index[0]); vn=w1.groupby('zrm').veh.agg(lambda s:s.value_counts().index[0])
m={}
for o,v in vo.items():
    c=vn[vn==v].index.tolist()
    if len(c)==1: m[o]=c[0]
print('old',len(old),'new',len(new),'mapped',len(m),'dup targets',pd.Series(m).duplicated().sum())
um=[o for o in old if o not in m]; print('unmapped old',[(o,z.loc[o,'last'].date(),z.loc[o,'n']) for o in um])
un=[n for n in new if n not in m.values()]; print('unmapped new',[(n,z.loc[n,'first'].date(),z.loc[n,'n']) for n in un])
# region by old prefix
reg={'L01':'Lublin','L02':'Biała Podlaska','L03':'Chełm','L04':'Zamość'}
team_region={o:reg.get(o[:3]) for o in old}
for o,n in m.items(): team_region[n]=reg.get(o[:3])
for n in un: team_region.setdefault(n,None)
tr=pd.Series(team_region); print(tr.value_counts(dropna=False))
# check new code hundreds digit vs region
chk=pd.DataFrame({'new':list(m.values()),'reg':[reg[o[:3]] for o in m]}); chk['h']=chk.new.str.extract(r'L01 (\d)\d\d')[0]; print(pd.crosstab(chk.h,chk.reg))
pd.Series(m).to_csv('out/team_old_new.csv'); tr.to_csv('out/team_region.csv')
# county -> region from home county
ch=d.drop_duplicates('zrm').set_index('zrm').county_home
print(pd.crosstab(ch.reindex(tr.index),tr).to_string())
