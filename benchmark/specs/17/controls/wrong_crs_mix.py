# Wrong: boroughs projected to EPSG:2263 (feet), crash points left in lon/lat, drawn on the same axes.
import geopandas as gpd, matplotlib.pyplot as plt
b = gpd.read_file('dataset/nyc_boroughs.geojson').to_crs(2263)     # projected (feet)
c = gpd.read_file('dataset/fatal_collisions.geojson')               # still lon/lat
fig, ax = plt.subplots()
b.plot(ax=ax, color='lightgrey'); ax.scatter(c.geometry.x, c.geometry.y, c='red', s=10)
plt.savefig('pred_results/collisions_map_vis.png')
