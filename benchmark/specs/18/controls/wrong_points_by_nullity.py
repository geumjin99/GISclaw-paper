# Wrong: plots the individual trees coloured by species nullity; no regional aggregation, no quadtree.
import geopandas as gpd, matplotlib.pyplot as plt
trees = gpd.read_file("dataset/street_trees_sample.geojson"); sfo = gpd.read_file("dataset/sfo_boroughs.geojson")
trees['species_missing'] = trees['Species'].isnull().astype(int)
ax = sfo.plot(color='white', edgecolor='k')
trees.plot(ax=ax, column='species_missing', markersize=1, cmap='coolwarm')     # no regional aggregation
plt.savefig("pred_results/trees_count_vis.png")
