# Correct: areal weighting in EPSG:6690; the per-municipality table is indexed by JISCODE instead
# of carrying it as a column. Multi-cell session code: the first cell uses `gpd` and `metro`
# from the session's layer-loading step.
metro = gpd.read_file('dataset/ShikokuMetropolitan.geojson')
print(f"Rows: {len(metro)}")
print(f"CRS: {metro.crs.to_string() if metro.crs is not None else None}")
print(f"Geometry types: { {k: int(c) for k, c in metro.geom_type.value_counts().items()} }")
print(f"Bounds (minx, miny, maxx, maxy): {[round(float(_b), 4) for _b in metro.total_bounds]}")
_cols = [c for c in metro.columns if c != metro.geometry.name]
print(f"Columns ({len(_cols)}): {_cols}")
import pandas as _pd
with _pd.option_context('display.max_columns', 30, 'display.width', 200, 'display.max_colwidth', 40):
    print(metro[_cols].head(3).to_string())


print(metro.AREATYPE.value_counts())
import geopandas as gpd
roads=gpd.read_file('dataset/AllSeasonRoads.geojson'); pop=gpd.read_file('dataset/ShikokuPopulation.geojson')
print(roads.crs, len(roads), roads.geom_type.value_counts()); print(pop.crs, len(pop), pop.geom_type.value_counts()); print(pop.head()); print(pop.D0001.describe())

crs=6690
rural=metro[metro.AREATYPE=='RURAL'].to_crs(crs)[['JISCODE','PNAME','AREANAME','geometry']]
rural['geometry']=rural.geometry.buffer(0)
p=pop.to_crs(crs); p['geometry']=p.geometry.buffer(0); p['orig_area']=p.area
r=roads.to_crs(crs)
# clip population to rural areas (keep rural area id)
rpop=gpd.overlay(p[['ID','D0001','orig_area','geometry']], rural, how='intersection')
rpop['pop']=rpop.D0001*rpop.area/rpop.orig_area
# buffer roads 2 km, dissolve, clip to rural
buf=gpd.GeoDataFrame(geometry=[r.buffer(2000).union_all()],crs=crs)
buf_rural=gpd.clip(buf, rural.union_all())
rpop['piece_area']=rpop.area
inb=gpd.overlay(rpop, buf_rural, how='intersection')
inb['pop_in']=inb['pop']*inb.area/inb.piece_area
tot=rpop.groupby('JISCODE')['pop'].sum(); acc=inb.groupby('JISCODE')['pop_in'].sum()
rural=rural.set_index('JISCODE'); rural['pop_total']=tot; rural['pop_access']=acc.reindex(rural.index).fillna(0)
rural['pct_access']=100*rural.pop_access/rural.pop_total
print(rural[['AREANAME','pop_total','pop_access','pct_access']].sort_values('pct_access').to_string())
print('Overall:',100*rural.pop_access.sum()/rural.pop_total.sum())

import matplotlib.pyplot as plt, os
os.makedirs('pred_results',exist_ok=True)
fig,ax=plt.subplots(figsize=(12,9))
metro.to_crs(crs).plot(ax=ax,color='lightgrey',edgecolor='white',linewidth=0.4)
rural.plot(column='pct_access',ax=ax,cmap='RdYlGn',edgecolor='black',linewidth=0.4,legend=True,
           scheme='NaturalBreaks',k=5,legend_kwds={'title':'% rural pop. within 2 km','loc':'lower right'})
r.plot(ax=ax,color='dimgray',linewidth=0.6,label='All-season roads')
ax.set_title(f'Shikoku: % of rural population within 2 km of an all-season road\n(grey = non-rural areas; overall rural access {100*rural.pop_access.sum()/rural.pop_total.sum():.1f}%)')
ax.set_axis_off(); plt.tight_layout(); plt.savefig('pred_results/accessibility.png',dpi=200); plt.close()
rural.drop(columns='geometry').to_csv('pred_results/rural_accessibility.csv')
print(os.listdir('pred_results'))