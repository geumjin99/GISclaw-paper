# Wrong: maps the elderly population count per block group; no temperature is interpolated.
import geopandas as gpd, matplotlib.pyplot as plt
b = gpd.read_file('dataset/block.geojson')
b['heat'] = b['Block_Groups_PopOver65']    # maps the elderly count, not temperature
b.plot(column='heat', cmap='OrRd', legend=True)
plt.savefig('pred_results/interpolated_urban_heat.png')
