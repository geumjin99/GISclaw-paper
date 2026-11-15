# Wrong: inspects all three factors but only the poverty layer is actually drawn with the service area.
import geopandas as gpd, matplotlib.pyplot as plt
bus = gpd.read_file('dataset/BusServiceArea.geojson')
dem = gpd.read_file('dataset/HamiltonDemographics.geojson').to_crs(bus.crs)
print(dem[['populationToPolygonSizeRating', 'households_ACSHHBPOV', 'AtRisk_ACSOVEH0']].describe())
fig, ax = plt.subplots(figsize=(10, 10))
dem.plot(ax=ax, column='households_ACSHHBPOV', cmap='Reds', alpha=0.5, legend=True)
bus.plot(ax=ax, color='none', edgecolor='black')
ax.set_title('Transit access and demographic need')
plt.savefig('pred_results/transit_access.png')
