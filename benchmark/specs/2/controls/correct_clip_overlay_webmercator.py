# Correct: geometric overlay (clip block groups to the service area) in Web Mercator,
# three separate transparent layers, literal use of the density rating column.
import geopandas as gpd
import matplotlib.pyplot as plt

bus = gpd.read_file('dataset/BusServiceArea.geojson').to_crs(3857)
dem = gpd.read_file('dataset/HamiltonDemographics.geojson').to_crs(3857)
inside = gpd.overlay(dem, bus[['geometry']], how='intersection')
fig, ax = plt.subplots(figsize=(10, 10))
dem.boundary.plot(ax=ax, color='lightgrey', linewidth=0.3)
inside.plot(ax=ax, column='populationToPolygonSizeRating', cmap='Greens', alpha=0.4)
inside.plot(ax=ax, column='households_ACSHHBPOV', cmap='Reds', alpha=0.4)
inside.plot(ax=ax, column='AtRisk_ACSOVEH0', cmap='Blues', alpha=0.4)
bus.boundary.plot(ax=ax, color='k')
ax.set_title('Demographics inside the bus service area')
plt.savefig('pred_results/transit_access.png')
