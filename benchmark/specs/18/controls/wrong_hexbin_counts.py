# Wrong: hexbin of the trees without C=: the map shows tree counts per hexagon, not the NULL-species share
# (the nullity flag is computed but never aggregated).
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
t = gpd.read_file('dataset/street_trees_sample.geojson').to_crs(32610)
city = gpd.read_file('dataset/sfo_boroughs.geojson').to_crs(32610)
t['null_flag'] = t['Species'].isnull().astype(int)
fig, ax = plt.subplots(figsize=(8, 8))
hb = ax.hexbin(t.geometry.x, t.geometry.y, gridsize=25, mincnt=1, cmap='Greens')
city.boundary.plot(ax=ax, color='k')
fig.colorbar(hb, label='trees')
fig.savefig('pred_results/trees_count_vis.png')
