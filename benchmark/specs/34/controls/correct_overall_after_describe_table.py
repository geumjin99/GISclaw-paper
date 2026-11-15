# Correct: areal weighting in EPSG:6672. The overall share (87.667 %) is printed on the same
# line as the two population totals it is the ratio of, appended to the last row of a
# describe() table, without any keyword.

import geopandas as gpd, numpy as np
m=gpd.read_file('dataset/ShikokuMetropolitan.geojson');r=gpd.read_file('dataset/AllSeasonRoads.geojson');p=gpd.read_file('dataset/ShikokuPopulation.geojson')
print(m.AREATYPE.value_counts(), m.crs==r.crs==p.crs)
m['geometry']=m.make_valid();p['geometry']=p.make_valid()
rural=m[m.AREATYPE=='RURAL'].copy();print(len(rural), rural.AREANAME.is_unique)



c=6672
rural=rural.to_crs(c).reset_index(drop=True);rural['rid']=rural.index
p=p.to_crs(c);r=r.to_crs(c)
pr=gpd.overlay(p[['D0001','geometry']].assign(parea=p.area),rural[['rid','geometry']],how='intersection')
pr['pop']=pr.D0001*pr.area/pr.parea
buf=gpd.GeoDataFrame(geometry=[r.buffer(2000).union_all()],crs=c)
pb=gpd.overlay(pr,buf,how='intersection')
pb['popb']=pb.D0001*pb.area/pb.parea
tot=pr.groupby('rid')['pop'].sum();acc=pb.groupby('rid')['popb'].sum()
rural['pop']=rural.rid.map(tot);rural['pop_buf']=rural.rid.map(acc).fillna(0)
rural['pct']=np.where(rural['pop']>0,rural.pop_buf/rural['pop']*100,np.nan)
print(rural[['pop','pop_buf','pct']].describe(), pr['pop'].sum(), pb.popb.sum(), pb.popb.sum()/pr['pop'].sum()*100)



import matplotlib.pyplot as plt, os
os.makedirs('pred_results',exist_ok=True)
fig,ax=plt.subplots(figsize=(11,8))
m.to_crs(c).plot(ax=ax,color='#eeeeee',edgecolor='#cccccc',lw=.3)
rural.plot(column='pct',cmap='YlGnBu',legend=True,ax=ax,edgecolor='k',lw=.3,legend_kwds={'label':'% rural population within 2 km of all-season road','shrink':.7})
r.plot(ax=ax,color='red',lw=.5)
ax.set_title('Rural population with access to all-season roads (2 km), Shikoku');ax.set_axis_off()
plt.savefig('pred_results/accessibility.png',dpi=200,bbox_inches='tight');plt.close()
rural.drop(columns='rid').to_file('pred_results/accessibility.geojson',driver='GeoJSON')
