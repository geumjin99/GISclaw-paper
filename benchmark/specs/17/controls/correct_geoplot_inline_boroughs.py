# Correct: geoplot pointplot over a polyplot of the boroughs read inline.
import geopandas as gpd, geoplot as gplt, geoplot.crs as gcrs, matplotlib.pyplot as plt
pts = gpd.read_file('dataset/fatal_collisions.geojson')
ax = gplt.polyplot(gpd.read_file('dataset/nyc_boroughs.geojson'), projection=gcrs.AlbersEqualArea(central_longitude=-74, central_latitude=40.7), facecolor='whitesmoke', edgecolor='grey', zorder=1)
gplt.pointplot(pts, ax=ax, hue='NUMBER OF PERSONS KILLED', cmap='Reds', legend=True, zorder=2)
ax.set_title('Fatal collisions in New York City, 2016 (n = 72)')
plt.savefig('pred_results/collisions_map_vis.png', dpi=130)
