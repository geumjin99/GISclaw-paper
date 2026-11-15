# Wrong: the nullity of PlantDate is mapped instead of the nullity of Species.
import geopandas as gpd, geoplot as gplt, geoplot.crs as gcrs, matplotlib.pyplot as plt
trees = gpd.read_file("dataset/street_trees_sample.geojson"); sfo = gpd.read_file("dataset/sfo_boroughs.geojson")
trees = trees.assign(nullity=trees["PlantDate"].isnull().astype(int))     # wrong attribute
ax = gplt.quadtree(trees, projection=gcrs.AlbersEqualArea(), hue="nullity", nmax=50, cmap="Greens", clip=sfo)
plt.savefig("pred_results/trees_count_vis.png")
