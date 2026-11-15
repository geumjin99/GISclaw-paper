# Wrong: select-by-location; population areas that INTERSECT the 2 km buffer are counted with their
# full population, which overstates the population within 2 km.
import geopandas as gpd, matplotlib.pyplot as plt
metro = gpd.read_file('dataset/ShikokuMetropolitan.geojson').to_crs(6690)
roads = gpd.read_file('dataset/AllSeasonRoads.geojson').to_crs(6690)
pop = gpd.read_file('dataset/ShikokuPopulation.geojson').to_crs(6690)
rural = metro[metro.AREATYPE == 'RURAL']
pts = pop.copy(); pts['geometry'] = pop.representative_point()
pop = pop.join(gpd.sjoin(pts, rural[['JISCODE', 'geometry']], predicate='within')['JISCODE'], how='inner')
buf = roads.buffer(2000).union_all()
pop['access'] = pop.intersects(buf)
g = pop.groupby('JISCODE').apply(lambda d: 100 * d.loc[d.access, 'D0001'].sum() / d.D0001.sum()).rename('pct_access').reset_index()
rural = rural.merge(g, on='JISCODE')
overall_pct = 100 * pop.loc[pop.access, 'D0001'].sum() / pop.D0001.sum()
print('overall', overall_pct)
ax = rural.plot(column='pct_access', legend=True); plt.savefig('pred_results/accessibility.png')
