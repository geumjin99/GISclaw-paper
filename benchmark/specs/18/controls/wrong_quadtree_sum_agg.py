# Wrong: geoplot quadtree with hue = null flag but agg=np.sum -> NUMBER of NULL trees per leaf, not a percentage.
import numpy as np, geopandas as gpd, geoplot as gplt, geoplot.crs as gcrs, matplotlib.pyplot as plt
trees = gpd.read_file("dataset/street_trees_sample.geojson")
sfo = gpd.read_file("dataset/sfo_boroughs.geojson")
trees["null_species"] = trees["Species"].isnull().astype(int)
ax = gplt.quadtree(trees, projection=gcrs.AlbersEqualArea(), hue="null_species", agg=np.sum, nmax=200,
                   cmap="Reds", legend=True, clip=sfo, edgecolor="white")
gplt.polyplot(sfo, facecolor="None", edgecolor="gray", ax=ax)
plt.savefig("pred_results/trees_count_vis.png")
