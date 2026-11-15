# Correct: population small areas allocated by representative point instead of areal weighting,
# buffer computed in the native CRS of the inputs (EPSG:4087).
import geopandas as gpd, matplotlib.pyplot as plt
metro = gpd.read_file('dataset/ShikokuMetropolitan.geojson')
roads = gpd.read_file('dataset/AllSeasonRoads.geojson')
pop = gpd.read_file('dataset/ShikokuPopulation.geojson')
rural = metro[metro.AREATYPE == 'RURAL']
pts = pop.copy(); pts['geometry'] = pop.representative_point()
pts = gpd.sjoin(pts, rural[['JISCODE', 'geometry']], predicate='within')
buf = roads.buffer(2000).union_all()
pts['access'] = pts.within(buf)
g = pts.groupby('JISCODE').apply(lambda d: d.loc[d.access, 'D0001'].sum() / d.D0001.sum())
rural = rural.merge(g.rename('share').reset_index(), on='JISCODE')
total_share = pts.loc[pts.access, 'D0001'].sum() / pts.D0001.sum()
ax = rural.plot(column='share', legend=True, cmap='viridis')
plt.savefig('pred_results/accessibility.png')
