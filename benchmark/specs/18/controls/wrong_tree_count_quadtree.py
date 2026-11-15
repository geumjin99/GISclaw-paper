# Wrong: the quadtree shows tree density only; the NULL-species percentage is computed city-wide and just printed.
import geopandas as gpd, geoplot as gplt, geoplot.crs as gcrs, matplotlib.pyplot as plt
trees = gpd.read_file("dataset/street_trees_sample.geojson"); sfo = gpd.read_file("dataset/sfo_boroughs.geojson")
null_pct = trees['Species'].isnull().mean() * 100          # computed but only printed
print(null_pct)
ax = gplt.quadtree(trees, projection=gcrs.AlbersEqualArea(), nmax=50, clip=sfo, edgecolor='white')   # tree density only
gplt.polyplot(sfo, ax=ax)
plt.savefig("pred_results/trees_count_vis.png")
