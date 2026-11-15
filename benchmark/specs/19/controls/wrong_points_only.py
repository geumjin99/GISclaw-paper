# Wrong: plots the sample points coloured by DO; no interpolated surface is produced.
import geopandas as gpd, matplotlib.pyplot as plt
pts = gpd.read_file('dataset/DissolvedO2.geojson'); bay = gpd.read_file('dataset/Bay.geojson')
ax = bay.plot(color='lightblue'); pts.plot(ax=ax, column='MeasureValue', markersize=2, legend=True)   # no interpolation
plt.savefig('pred_results/interploated_water_quality.png')
