# Wrong: maps the three demographic factors but never brings in the bus service area.
import geopandas as gpd
import matplotlib.pyplot as plt

dem = gpd.read_file('dataset/HamiltonDemographics.geojson')
fig, ax = plt.subplots(figsize=(10, 10))
dem.plot(ax=ax, column='populationToPolygonSizeRating', cmap='Greens', alpha=0.4)
dem.plot(ax=ax, column='households_ACSHHBPOV', cmap='Reds', alpha=0.4)
dem.plot(ax=ax, column='AtRisk_ACSOVEH0', cmap='Blues', alpha=0.4)
plt.savefig('pred_results/transit_access.png')
