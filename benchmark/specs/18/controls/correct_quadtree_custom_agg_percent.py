# Correct: geoplot.quadtree(hue=<0/1 NULL flag>, agg=<user function returning 100*sum/len>) -
# the NULL share in percent.
import numpy as np, geopandas as gpd, geoplot, matplotlib.pyplot as plt
trees = gpd.read_file("dataset/street_trees_sample.geojson")
gdf_trees = trees.assign(species_null=trees["Species"].isnull().astype(int))

def pct_null(series):
    if len(series) == 0:
        return np.nan
    return 100.0 * np.sum(series) / len(series)

ax = geoplot.quadtree(gdf_trees, hue='species_null', agg=pct_null, cmap='OrRd', legend=True, nmin=10)
plt.savefig('pred_results/trees_count_vis.png', dpi=100, bbox_inches='tight')
