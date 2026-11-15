# Wrong: boroughs are loaded but never drawn; the map shows the
# crash points on an empty background, no administrative regions.
import geopandas as gpd, geoplot as gplt, geoplot.crs as gcrs, matplotlib.pyplot as plt
boroughs = gpd.read_file('dataset/nyc_boroughs.geojson')
crashes = gpd.read_file('dataset/fatal_collisions.geojson')
ax = gplt.pointplot(crashes, projection=gcrs.AlbersEqualArea(), hue='BOROUGH', scale='NUMBER OF PERSONS KILLED', limits=(6, 18), legend=True)
ax.set_title('Fatal collisions in NYC, 2016')
plt.savefig('pred_results/collisions_map_vis.png')
