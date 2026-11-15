# Correct: areal weighting in UTM 53N (EPSG:32653) after repairing invalid geometries; the overall
# share is printed next to the two totals ('... of ... -> 87.67 %', 'overall access 87.67%').

import geopandas as gpd, pandas as pd
from shapely.validation import make_valid
towns0 = gpd.read_file('dataset/ShikokuMetropolitan.geojson')
roads0 = gpd.read_file('dataset/AllSeasonRoads.geojson')
pop0 = gpd.read_file('dataset/ShikokuPopulation.geojson')
def fix(g):
    g = g.copy()
    bad = ~g.geometry.is_valid
    print('invalid before:', bad.sum())
    g.loc[bad,'geometry'] = g.loc[bad,'geometry'].apply(make_valid)
    # keep only polygonal parts if make_valid produced collections
    def poly_only(geom):
        if geom.geom_type == 'GeometryCollection':
            from shapely.geometry import MultiPolygon
            parts=[p for p in geom.geoms if p.geom_type in ('Polygon','MultiPolygon')]
            return gpd.GeoSeries(parts).union_all() if parts else geom
        return geom
    g['geometry'] = g.geometry.apply(poly_only)
    return g
towns = fix(towns0).to_crs(32653)
pop = fix(pop0).to_crs(32653)
roads = roads0.to_crs(32653)
for n,g in [('towns',towns),('roads',roads),('pop',pop)]:
    print(n, len(g), g.crs.to_epsg(), 'invalid:', (~g.is_valid).sum(), g.geom_type.value_counts().to_dict())
print(towns['AREATYPE'].value_counts())
print('Total D0001:', pop['D0001'].sum(), 'missing:', pop['D0001'].isna().sum())



rural = towns[towns['AREATYPE']=='RURAL'].copy()
rural['area_key'] = rural['OBJECTID'].astype(str) + '_' + rural['AREANAME']
print(len(rural), rural['area_key'].is_unique, rural.geom_type.value_counts().to_dict(), pop.geom_type.value_counts().to_dict())
pop_s = pop[['ID','D0001','geometry']].copy()
pop_s['orig_area'] = pop_s.geometry.area
rural_pop = gpd.overlay(pop_s, rural[['area_key','AREANAME','geometry']], how='intersection', keep_geom_type=True)
rural_pop['piece_area'] = rural_pop.geometry.area
rural_pop['pop_w'] = rural_pop['D0001'] * rural_pop['piece_area'] / rural_pop['orig_area']
rural_tot = rural_pop.groupby('area_key')['pop_w'].sum()
print(len(rural_pop), 'pieces;', rural_pop['ID'].nunique(), 'pop polygons matched')
print('rural areas with pop:', len(rural_tot), 'of', len(rural))
print('Total rural population: %.0f' % rural_tot.sum())
print(rural_tot.describe())


print(rural_pop.columns.tolist()); print(rural_pop.head(3)); print(rural_tot.sum(), rural.columns.tolist())

import geopandas as gpd
from shapely.ops import unary_union
road_buf_geom = unary_union(roads.geometry.buffer(2000))
road_buf = gpd.GeoDataFrame({'id':[1]}, geometry=[road_buf_geom], crs=roads.crs)
print('buffer area km2:', road_buf.area.sum()/1e6)
rural_buf = gpd.overlay(rural[['area_key','AREANAME','geometry']], road_buf[['geometry']], how='intersection', keep_geom_type=True)
print('rural_buf pieces:', len(rural_buf), 'area km2:', rural_buf.area.sum()/1e6, 'rural area km2:', rural.area.sum()/1e6)
buf_pop = gpd.overlay(rural_pop[['ID','D0001','orig_area','area_key','geometry']], road_buf[['geometry']], how='intersection', keep_geom_type=True)
buf_pop['buf_area'] = buf_pop.area
buf_pop['pop_w'] = buf_pop['D0001'] * buf_pop['buf_area'] / buf_pop['orig_area']
print('buf_pop pieces:', len(buf_pop), 'areas covered:', buf_pop.area_key.nunique())
print('rural pop within 2km:', buf_pop.pop_w.sum(), 'of', rural_tot.sum(), '->', 100*buf_pop.pop_w.sum()/rural_tot.sum(), '%')
print('max piece/orig ratio:', (buf_pop.buf_area/buf_pop.orig_area).max())

print(rural.columns.tolist()); print(rural_pop.columns.tolist()); print(buf_pop.columns.tolist())
print(rural_tot.head()); print(rural_tot.name, rural_tot.index.name, len(rural_tot), rural_tot.sum())

import numpy as np, pandas as pd
buf_tot = buf_pop.groupby('area_key')['pop_w'].sum().rename('pop_in_buffer')
summ = pd.concat([rural_tot.rename('rural_pop'), buf_tot], axis=1)
summ['pop_in_buffer'] = summ['pop_in_buffer'].fillna(0)
summ['pct_access'] = np.where(summ['rural_pop']>0, 100*summ['pop_in_buffer']/summ['rural_pop'], np.nan)
print("areas:", len(summ), "with zero rural pop:", (summ.rural_pop<=0).sum(), "with no buffer:", (summ.pop_in_buffer==0).sum())
rural = rural.drop(columns=[c for c in ['rural_pop','pop_in_buffer','pct_access'] if c in rural.columns])
rural = rural.merge(summ, left_on='area_key', right_index=True, how='left')
print("missing after merge:", rural[['rural_pop','pct_access']].isna().sum().to_dict())
tot_r, tot_b = rural.rural_pop.sum(), rural.pop_in_buffer.sum()
print(f"Total rural pop {tot_r:,.0f}; within 2 km {tot_b:,.0f}; overall access {100*tot_b/tot_r:.2f}%")
print(rural.pct_access.describe()[['min','max','mean']])
pd.set_option('display.width',200)
print(rural[['area_key','PNAME','AREANAME','rural_pop','pop_in_buffer','pct_access']].sort_values('pct_access').round(1).to_string(index=False))

import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.makedirs('pred_results', exist_ok=True)
print(rural[['AREANAME','rural_pop','pop_in_buffer','pct_access']].describe())
print(rural['pct_access'].isna().sum())
fig, ax = plt.subplots(figsize=(12,9))
towns.plot(ax=ax, color='lightgrey', edgecolor='white', linewidth=0.4)
rural.plot(ax=ax, column='pct_access', cmap='YlGnBu', edgecolor='black', linewidth=0.4, legend=True,
           missing_kwds={'color':'white','hatch':'///','label':'No population'},
           legend_kwds={'label':'% rural population within 2 km of all-season roads','shrink':0.6})
roads.plot(ax=ax, color='darkred', linewidth=0.5, alpha=0.7)
ax.set_title(f'Access to all-season roads in rural Shikoku (2015 population)\nOverall: {100*tot_b/tot_r:.1f}% of rural population within 2 km', fontsize=13)
ax.set_axis_off()
plt.tight_layout()
plt.savefig('pred_results/accessibility.png', dpi=200)
plt.close()
print(os.listdir('pred_results'))