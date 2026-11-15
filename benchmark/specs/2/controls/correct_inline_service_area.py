# Correct: density derived, three transparent layers, service area read inline in the plot call.
import geopandas as gpd, matplotlib.pyplot as plt
d = gpd.read_file('dataset/HamiltonDemographics.geojson').to_crs(3857)
d['dens'] = d['populationtotals_TOTPOP_CY'] / (d.to_crs(32616).area / 1e6)
fig, ax = plt.subplots(figsize=(11, 11))
for col, cm in (('dens', 'Greens'), ('households_ACSHHBPOV', 'Oranges'), ('AtRisk_ACSOVEH0', 'Purples')):
    d.plot(ax=ax, column=col, cmap=cm, alpha=0.35, scheme='quantiles', k=5)
gpd.read_file('dataset/BusServiceArea.geojson').to_crs(3857).plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5)
ax.set_title('Bus service area over density, poverty and no-vehicle households')
plt.savefig('pred_results/transit_access.png', dpi=150)
