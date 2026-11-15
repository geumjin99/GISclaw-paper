# Wrong: computes the share of each rural municipality's AREA inside the 2 km buffer; the task
# asks for the share of the rural population.
import geopandas as gpd, matplotlib.pyplot as plt
metro = gpd.read_file('dataset/ShikokuMetropolitan.geojson').to_crs(6690)
roads = gpd.read_file('dataset/AllSeasonRoads.geojson').to_crs(6690)
rural = metro[metro.AREATYPE == 'RURAL'].copy()
buf = roads.buffer(2000).union_all()
rural['pct'] = 100 * rural.intersection(buf).area / rural.area     # share of AREA, not population
overall = 100 * rural.intersection(buf).area.sum() / rural.area.sum()
rural.plot(column='pct', legend=True)
plt.savefig('pred_results/accessibility.png')
