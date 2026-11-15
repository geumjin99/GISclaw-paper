# Wrong: buffers the roads by 2 map units (metres) instead of the 2 km the task asks for.
import geopandas as gpd, matplotlib.pyplot as plt
metro = gpd.read_file('dataset/ShikokuMetropolitan.geojson')
roads = gpd.read_file('dataset/AllSeasonRoads.geojson')
pop = gpd.read_file('dataset/ShikokuPopulation.geojson')
rural = metro[metro.AREATYPE == 'RURAL']
pop['a0'] = pop.area
p = gpd.overlay(pop[['D0001', 'a0', 'geometry']], rural[['JISCODE', 'geometry']], how='intersection')
p['pop'] = p.D0001 * p.area / p.a0
buf = gpd.GeoDataFrame(geometry=[roads.buffer(2).union_all()], crs=roads.crs)   # "2 km" typed as 2 (metres)
b = gpd.overlay(p, buf, how='intersection'); b['popb'] = b.D0001 * b.area / b.a0
res = rural.merge((100 * b.groupby('JISCODE').popb.sum() / p.groupby('JISCODE')['pop'].sum()).rename('pct').reset_index(), on='JISCODE')
overall = 100 * b.popb.sum() / p['pop'].sum()
res.plot(column='pct', legend=True)
plt.savefig('pred_results/accessibility.png')
