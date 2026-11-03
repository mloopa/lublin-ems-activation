import pandas as pd, numpy as np, json
d=pd.read_parquet('analytic_v9.parquet')
m=pd.read_csv('out/team_old_new.csv',index_col=0).iloc[:,0].to_dict()
m.update({'L01 11':'L01 203','L02 06':'L01 306','L02 07':'L01 307','L02 10':'L01 310','L02 20':'L01 320','L03 04':'L01 324','L04 18':'L01 422'})
d['team']=d.zrm.map(lambda z:m.get(z,z))
c=pd.read_parquet('calls2.parquet',columns=['zrm','county']); c['team']=c.zrm.map(lambda z:m.get(z,z))
home=c.dropna(subset=['county']).groupby('team').county.agg(lambda s:s.value_counts().index[0])
print('teams',d.team.nunique()); wu=[t for t in home.index if t in ('L01 336',)]; print('override',wu,home.get('L01 336'))
home['L01 336']='powiat włodawski'
d['county_station']=d.team.map(home)
reg_of={'powiat Biała Podlaska':'Biała Podlaska','powiat bialski':'Biała Podlaska','powiat parczewski':'Biała Podlaska','powiat radzyński':'Biała Podlaska','powiat łukowski':'Biała Podlaska',
 'powiat Chełm':'Chełm','powiat chełmski':'Chełm','powiat krasnostawski':'Chełm','powiat włodawski':'Chełm',
 'powiat Zamość':'Zamość','powiat biłgorajski':'Zamość','powiat hrubieszowski':'Zamość','powiat janowski':'Zamość','powiat tomaszowski':'Zamość','powiat zamojski':'Zamość'}
d['region']=d.county_station.map(lambda x:reg_of.get(x,'Lublin'))
print(d.groupby('region').team.nunique())
# recompute daily teams and occupancy denominators
teams=d.groupby('date').team.nunique(); d['teams_day']=d.date.map(teams); d['busy_frac']=d.busy_reg/d.teams_day
tc=d.groupby(['county_station','date']).team.nunique(); d['teams_cty']=pd.MultiIndex.from_frame(d[['county_station','date']]).map(tc)
d['busy_frac_cty']=(d.busy_cty/d.teams_cty).clip(0,1.5)
d['county_home']=d.county_station
print('teams/day',teams.mean().round(1),teams.min(),teams.max())
d.to_parquet('analytic.parquet')
json.dump({'teams_mean':float(teams.mean()),'teams_min':int(teams.min()),'teams_max':int(teams.max()),'n_teams':int(d.team.nunique())},open('out/teams.json','w'))
