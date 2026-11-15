# Wrong: boroughs drawn in one figure, geoplot
# pointplot opens a NEW figure; the points figure is saved as another file and the required
# collisions_map_vis.png holds the boroughs only.
import geopandas as gpd, geoplot, matplotlib.pyplot as plt
b = gpd.read_file('dataset/nyc_boroughs.geojson').to_crs(epsg=3857)
c = gpd.read_file('dataset/fatal_collisions.geojson').to_crs(epsg=3857)
b.plot(column='BoroName', cmap='viridis', legend=True)
ax = geoplot.pointplot(c, hue='NUMBER OF PERSONS KILLED', legend=True)
plt.savefig('pred_results/choropleth_map_point_plot.png', dpi=150, bbox_inches='tight')
plt.close()
plt.savefig('pred_results/collisions_map_vis.png', dpi=150, bbox_inches='tight')
plt.close()
