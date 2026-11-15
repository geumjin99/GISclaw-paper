# Wrong: each factor is drawn as a graded layer, but in its own PNG; transit_access.png shows only
# the service area over the block-group outlines - the requested overlay is not in the requested map.
import geopandas as gpd
import matplotlib.pyplot as plt

bus = gpd.read_file('dataset/BusServiceArea.geojson')
dem = gpd.read_file('dataset/HamiltonDemographics.geojson').to_crs(bus.crs)
inside = gpd.overlay(dem, bus[['geometry']], how='intersection')
for col, cmap, fn in (('households_ACSHHBPOV', 'Reds', 'poverty_map.png'),
                      ('AtRisk_ACSOVEH0', 'Blues', 'vehicle_map.png'),
                      ('populationToPolygonSizeRating', 'Greens', 'density_map.png')):
    fig, ax = plt.subplots(figsize=(8, 8))
    inside.plot(ax=ax, column=col, cmap=cmap, alpha=0.6)
    plt.savefig('pred_results/' + fn)
    plt.close()
fig, ax = plt.subplots(figsize=(8, 8))
dem.boundary.plot(ax=ax, color='lightgrey', linewidth=0.3)
bus.plot(ax=ax, color='orange', alpha=0.4)
plt.savefig('pred_results/transit_access.png')
