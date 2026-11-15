# Correct: MTM zone 10 (EPSG:2952), 2.5 km buffers of all stations, gap computed with shapely
# and plotted directly (no GeoDataFrame for the gap).
import geopandas as gpd, matplotlib.pyplot as plt
stations = gpd.read_file('dataset/fire_station.geojson').to_crs(2952)
district = gpd.read_file('dataset/etobicoke.geojson').to_crs(2952)
covered = stations.buffer(2500).union_all()
uncovered = district.union_all().difference(covered)
fig, ax = plt.subplots(figsize=(10, 10))
district.plot(ax=ax, color='#eeeeee', edgecolor='grey')
gpd.GeoSeries([covered.intersection(district.union_all())], crs=2952).plot(ax=ax, color='lightsteelblue')
gpd.GeoSeries([uncovered], crs=2952).plot(ax=ax, color='crimson')
stations.plot(ax=ax, color='k', markersize=6)
ax.set_xlim(*district.total_bounds[[0, 2]]); ax.set_ylim(*district.total_bounds[[1, 3]])
ax.set_title(f'Etobicoke fire coverage gaps: {uncovered.area/1e6:.2f} km2 beyond 2.5 km')
plt.savefig('pred_results/Fire_Service_Analysis.png')
