# Wrong: service area replaced by its centroids buffered 50 m (only "stops"), so service coverage is
# essentially absent; a full-coverage clipped demographic layer exists in memory but is never shown.
import geopandas as gpd, matplotlib.pyplot as plt
bus = gpd.read_file('dataset/BusServiceArea.geojson').to_crs(32616)
dem = gpd.read_file('dataset/HamiltonDemographics.geojson').to_crs(32616)
inside = gpd.clip(dem, bus)          # computed but never shown
stops = bus.copy(); stops['geometry'] = bus.geometry.centroid.buffer(50)
fig, ax = plt.subplots(figsize=(10, 10))
for col, cm in (('populationToPolygonSizeRating', 'Greens'), ('households_ACSHHBPOV', 'Reds'), ('AtRisk_ACSOVEH0', 'Blues')):
    dem.plot(ax=ax, column=col, cmap=cm, alpha=0.35)
stops.plot(ax=ax, color='k')
plt.savefig('pred_results/transit_access.png')
