# Wrong: the task prescribes geoplot ("plot ... on a map by geoplot"); this map is drawn with
# cartopy and matplotlib and uses no geoplot call.
import geopandas as gpd, matplotlib.pyplot as plt, cartopy.crs as ccrs
b = gpd.read_file('dataset/nyc_boroughs.geojson'); c = gpd.read_file('dataset/fatal_collisions.geojson')
ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-74, central_latitude=40.7))
ax.add_geometries(b.geometry, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='k')
for name, grp in c.groupby('BOROUGH'):
    ax.scatter(grp.geometry.x, grp.geometry.y, s=20 * grp['NUMBER OF PERSONS KILLED'], label=name, transform=ccrs.PlateCarree())
ax.set_extent([-74.3, -73.65, 40.45, 40.95], crs=ccrs.PlateCarree()); ax.legend()
plt.savefig('pred_results/collisions_map_vis.png')
