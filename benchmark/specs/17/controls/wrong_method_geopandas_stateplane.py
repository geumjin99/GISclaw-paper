# Wrong: the task prescribes geoplot ("plot ... on a map by geoplot"); this map is drawn with
# geopandas in State Plane coordinates and uses no geoplot call.
import geopandas as gpd, matplotlib.pyplot as plt
b = gpd.read_file('dataset/nyc_boroughs.geojson').to_crs(2263)
c = gpd.read_file('dataset/fatal_collisions.geojson').to_crs(2263)
ax = b.plot(color='whitesmoke', edgecolor='grey', figsize=(8, 8))
c.plot(ax=ax, column='BOROUGH', markersize=c['NUMBER OF PERSONS KILLED'] * 30, legend=True, cmap='Set1')
ax.set_title('Fatal collisions NYC 2016'); ax.set_axis_off()
plt.savefig('pred_results/collisions_map_vis.png', dpi=120)
