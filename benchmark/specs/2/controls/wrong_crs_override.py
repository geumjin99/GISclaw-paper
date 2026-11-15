# Wrong: projects the demographics to UTM but *overrides* (set_crs) the service-area CRS
# instead of reprojecting it, so the service area lands near (0, 0) in metres.
import geopandas as gpd
import matplotlib.pyplot as plt

dem = gpd.read_file('dataset/HamiltonDemographics.geojson').to_crs(32616)
bus = gpd.read_file('dataset/BusServiceArea.geojson').set_crs(32616, allow_override=True)
served = gpd.overlay(dem, bus[['geometry']], how='union')
fig, ax = plt.subplots(figsize=(10, 10))
dem.plot(ax=ax, column='populationToPolygonSizeRating', cmap='Greens', alpha=0.4)
dem.plot(ax=ax, column='households_ACSHHBPOV', cmap='Reds', alpha=0.4)
dem.plot(ax=ax, column='AtRisk_ACSOVEH0', cmap='Blues', alpha=0.4)
bus.boundary.plot(ax=ax, color='k')
plt.savefig('pred_results/transit_access.png')
