# Wrong: no interpolation; each block group gets the mean of the stations inside it, so most
# block groups have no value.
import geopandas as gpd, matplotlib.pyplot as plt
t = gpd.read_file('dataset/Temperature.geojson'); b = gpd.read_file('dataset/block.geojson')
j = gpd.sjoin(t, b, predicate='within')
m = j.groupby('index_right')['TemperatureF'].mean()
b['mean_temp'] = b.index.map(m)            # most block groups get no value (no station inside)
b.plot(column='mean_temp', cmap='OrRd', legend=True, missing_kwds={'color': 'lightgrey'})
plt.savefig('pred_results/interpolated_urban_heat.png')
