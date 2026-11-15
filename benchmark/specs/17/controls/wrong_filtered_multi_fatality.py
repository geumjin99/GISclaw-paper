# Wrong: keeps only crashes with more than one person killed, so most fatal crashes are missing.
import geopandas as gpd, geoplot as gplt, geoplot.crs as gcrs, matplotlib.pyplot as plt
b = gpd.read_file('dataset/nyc_boroughs.geojson'); c = gpd.read_file('dataset/fatal_collisions.geojson')
c2016 = c[c['NUMBER OF PERSONS KILLED'] > 1]          # misreads 'fatal' as multi-fatality crashes
ax = gplt.polyplot(b, projection=gcrs.AlbersEqualArea())
gplt.pointplot(c2016, ax=ax, hue='BOROUGH')
plt.savefig('pred_results/collisions_map_vis.png')
