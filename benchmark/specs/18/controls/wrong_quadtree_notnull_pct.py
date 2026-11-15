# Wrong: quadtree with hue = Species.notnull()*100 (complement, in percent): maps the share WITH species,
# not the NULL share the instruction asks for.
import numpy as np, geopandas as gpd, geoplot as gplt, geoplot.crs as gcrs, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
trees = gpd.read_file("dataset/street_trees_sample.geojson")
sfo = gpd.read_file("dataset/sfo_boroughs.geojson")
t = trees.assign(nullity=trees["Species"].notnull().astype(int) * 100)
ax = gplt.quadtree(t, projection=gcrs.AlbersEqualArea(), hue="nullity", nmax=100, agg=np.mean, cmap="Reds",
                   clip=sfo.geometry, edgecolor="white", linewidth=0.5, legend=True)
plt.savefig("pred_results/trees_count_vis.png", dpi=150, bbox_inches="tight")
