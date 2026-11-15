# Wrong: geoplot.quadtree with a user-defined agg that returns the NUMBER of
# NULL-species trees per cell, not their share.
import numpy as np, geopandas as gpd, geoplot, matplotlib.pyplot as plt
trees = gpd.read_file("dataset/street_trees_sample.geojson")
gdf_trees = trees.assign(species_null=trees["Species"].isnull().astype(int))

def n_null(series):
    return float(np.sum(series))

ax = geoplot.quadtree(gdf_trees, hue='species_null', agg=n_null, cmap='OrRd', legend=True, nmin=10)
plt.savefig('pred_results/trees_count_vis.png', dpi=100, bbox_inches='tight')
