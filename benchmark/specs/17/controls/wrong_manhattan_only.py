# Wrong: maps only the Manhattan crashes over the Manhattan polygon instead of all of New York City.
import geopandas as gpd, matplotlib.pyplot as plt
b = gpd.read_file('dataset/nyc_boroughs.geojson'); c = gpd.read_file('dataset/fatal_collisions.geojson')
m = c[c.BOROUGH == 'MANHATTAN']
ax = b[b.BoroName == 'Manhattan'].plot(color='lightgrey'); m.plot(ax=ax, color='red')
plt.savefig('pred_results/collisions_map_vis.png')
