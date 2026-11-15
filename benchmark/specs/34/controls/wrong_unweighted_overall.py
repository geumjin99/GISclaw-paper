# Wrong: per-municipality shares are right (areal weighting), but the reported overall share is the
# unweighted mean of municipalities instead of the population-weighted share.
import geopandas as gpd, matplotlib.pyplot as plt
metro = gpd.read_file('dataset/ShikokuMetropolitan.geojson').to_crs(6690)
roads = gpd.read_file('dataset/AllSeasonRoads.geojson').to_crs(6690)
pop = gpd.read_file('dataset/ShikokuPopulation.geojson').to_crs(6690)
pop['a0'] = pop.area
rural = metro[metro.AREATYPE == 'RURAL'][['JISCODE', 'geometry']]
rp = gpd.overlay(pop[['D0001', 'a0', 'geometry']], rural, how='intersection')
rp['pop'] = rp.D0001 * rp.area / rp.a0
buf = gpd.GeoDataFrame(geometry=[roads.buffer(2000).union_all()], crs=6690)
ib = gpd.overlay(rp, buf, how='intersection'); ib['pop_in'] = ib.D0001 * ib.area / ib.a0
tab = rp.groupby('JISCODE')['pop'].sum().to_frame().join(ib.groupby('JISCODE')['pop_in'].sum()).fillna(0).reset_index()
tab['share'] = 100 * tab.pop_in / tab['pop']
rural = rural.merge(tab, on='JISCODE')
rural_access_pct = tab.share.mean()
print('percentage of rural population with access:', rural_access_pct)
open('pred_results/result.txt', 'w').write(f'Rural population within 2 km of all-season roads: {rural_access_pct:.1f}%\n')
ax = rural.plot(column='share', legend=True); plt.savefig('pred_results/accessibility.png')
