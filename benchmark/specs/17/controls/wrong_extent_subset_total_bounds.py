# Wrong: geoplot map whose extent is the total_bounds of Manhattan only; crashes in the other
# boroughs fall outside the saved map.
import geopandas as gpd, geoplot as gplt, geoplot.crs as gcrs, matplotlib.pyplot as plt
b = gpd.read_file('dataset/nyc_boroughs.geojson')
c = gpd.read_file('dataset/fatal_collisions.geojson')
man = b[b['BoroName'] == 'Manhattan']
ax = gplt.polyplot(b, projection=gcrs.AlbersEqualArea(), facecolor='lightgray', edgecolor='white')
gplt.pointplot(c, ax=ax, color='red', scale='NUMBER OF PERSONS KILLED', limits=(5, 12), extent=man.total_bounds)
plt.savefig('pred_results/collisions_map_vis.png', dpi=150, bbox_inches='tight')
