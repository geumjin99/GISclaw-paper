# Wrong: no interpolation surface; each block group gets the temperature of its single nearest station.
import geopandas as gpd, matplotlib.pyplot as plt
t = gpd.read_file('dataset/Temperature.geojson'); b = gpd.read_file('dataset/block.geojson')
cent = b.copy(); cent['geometry'] = b.geometry.centroid
nn = gpd.sjoin_nearest(cent, t[['TemperatureF', 'geometry']], how='left')
nn = nn[~nn.index.duplicated()]
b['avg_temp'] = nn['TemperatureF']
ax = b.plot(column='avg_temp', cmap='OrRd', legend=True, figsize=(9, 9))
b[(b.avg_temp > 81) & (b.Block_Groups_Over65Density > 100000)].boundary.plot(ax=ax, color='blue')
plt.savefig('pred_results/interpolated_urban_heat.png')
