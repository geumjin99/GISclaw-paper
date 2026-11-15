# Wrong: loads and dissolves the service area but the map only shows the three demographic layers.
import geopandas as gpd, matplotlib.pyplot as plt
bus = gpd.read_file('dataset/BusServiceArea.geojson')
dem = gpd.read_file('dataset/HamiltonDemographics.geojson').to_crs(bus.crs)
bus_d = bus.dissolve()
fig, ax = plt.subplots(figsize=(10, 10))
dem.plot(ax=ax, column='populationToPolygonSizeRating', cmap='Greens', alpha=0.4)
dem.plot(ax=ax, column='households_ACSHHBPOV', cmap='Reds', alpha=0.4)
dem.plot(ax=ax, column='AtRisk_ACSOVEH0', cmap='Blues', alpha=0.4)
ax.set_title('Demographic need, Hamilton County')
plt.savefig('pred_results/transit_access.png')
