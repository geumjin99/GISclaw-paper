# Wrong: no rural filter; the access share is computed for all municipalities, urban ones included.
import geopandas as gpd, matplotlib.pyplot as plt
metro = gpd.read_file('dataset/ShikokuMetropolitan.geojson').to_crs(6690)
roads = gpd.read_file('dataset/AllSeasonRoads.geojson').to_crs(6690)
pop = gpd.read_file('dataset/ShikokuPopulation.geojson').to_crs(6690)
pop['a0'] = pop.area
p = gpd.overlay(pop[['D0001', 'a0', 'geometry']], metro[['JISCODE', 'geometry']], how='intersection')  # all areas
p['pop'] = p.D0001 * p.area / p.a0
buf = gpd.GeoDataFrame(geometry=[roads.buffer(2000).union_all()], crs=6690)
b = gpd.overlay(p, buf, how='intersection'); b['popb'] = b.D0001 * b.area / b.a0
res = metro.merge((100 * b.groupby('JISCODE').popb.sum() / p.groupby('JISCODE')['pop'].sum()).rename('pct').reset_index(), on='JISCODE')
overall = 100 * b.popb.sum() / p['pop'].sum()
res.plot(column='pct', legend=True)
plt.savefig('pred_results/accessibility.png')
