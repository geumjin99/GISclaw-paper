# Correct: geoplot quadtree with the NULL-species flag as hue (mean aggregation).
import geopandas as gpd, geoplot as gplt, geoplot.crs as gcrs, matplotlib.pyplot as plt
trees = gpd.read_file("dataset/street_trees_sample.geojson")
sfo = gpd.read_file("dataset/sfo_boroughs.geojson")
trees_n = trees.assign(species_null=trees["Species"].isnull().astype(int))
ax = gplt.quadtree(trees_n, projection=gcrs.AlbersEqualArea(), hue="species_null", nmax=100, cmap="Reds",
                   scheme="Quantiles", legend=True, clip=sfo, edgecolor="white", linewidth=1)
gplt.polyplot(sfo, facecolor="None", edgecolor="gray", linewidth=1, zorder=2, ax=ax)
plt.savefig("pred_results/trees_count_vis.png")
