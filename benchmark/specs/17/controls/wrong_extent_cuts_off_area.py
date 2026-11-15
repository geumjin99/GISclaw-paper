# Wrong: geoplot map with an extent that cuts off Staten Island and the eastern/southern edges, so a part
# of the 2016 crashes is not visible on the saved map.
import geopandas as gpd, geoplot as gplt, geoplot.crs as gcrs, matplotlib.pyplot as plt
boroughs = gpd.read_file('dataset/nyc_boroughs.geojson')
crashes = gpd.read_file('dataset/fatal_collisions.geojson')
ext = (-74.03, 40.62, -73.80, 40.90)
ax = gplt.polyplot(boroughs, projection=gcrs.WebMercator(), extent=ext, facecolor='lightgrey', edgecolor='white')
gplt.pointplot(crashes, ax=ax, extent=ext, color='red', scale='NUMBER OF PERSONS KILLED', limits=(5, 15))
ax.set_title('Fatal collisions in NYC, 2016')
plt.savefig('pred_results/collisions_map_vis.png')
print('visible', ((crashes.geometry.x.between(ext[0], ext[2])) & (crashes.geometry.y.between(ext[1], ext[3]))).sum())
