# Wrong: guesses the fields - total population as "poverty", water area as "no vehicle access".
import geopandas as gpd
import matplotlib.pyplot as plt

bus = gpd.read_file('dataset/BusServiceArea.geojson')
dem = gpd.read_file('dataset/HamiltonDemographics.geojson')
fig, ax = plt.subplots(figsize=(10, 10))
dem.plot(ax=ax, column='populationToPolygonSizeRating', cmap='Greens', alpha=0.4)
dem.plot(ax=ax, column='populationtotals_TOTPOP_CY', cmap='Reds', alpha=0.4)
dem.plot(ax=ax, column='AREAWATER', cmap='Blues', alpha=0.4)
bus.boundary.plot(ax=ax, color='k')
plt.savefig('pred_results/transit_access.png')
