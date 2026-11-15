# Wrong: maps notnull() (share of trees WITH a species); the instruction asks for the NULL share.
# Adapted from the reference script of GeoAnalystBench (Zhang et al. 2025, Apache-2.0).
import geopandas as gpd, geoplot as gplt, geoplot.crs as gcrs, matplotlib.pyplot as plt
trees = gpd.read_file("dataset/street_trees_sample.geojson")
sfo = gpd.read_file("dataset/sfo_boroughs.geojson")
trees_n = trees.assign(nullity=trees["Species"].notnull().astype(int))
ax = gplt.quadtree(trees_n, projection=gcrs.AlbersEqualArea(), hue="nullity", nmax=1, cmap="Greens",
                   scheme="Quantiles", legend=True, clip=sfo, edgecolor="white", linewidth=1)
gplt.polyplot(sfo, facecolor="None", edgecolor="gray", linewidth=1, zorder=2, ax=ax)
plt.savefig("pred_results/trees_count_vis.png")
