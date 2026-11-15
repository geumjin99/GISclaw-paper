"""Reference solution for task 18: share of street trees with missing (NULL) species, by quadtree region, San Francisco.

Steps: per tree, species_null = Species is NULL (652 of 9 878 = 6.6 %); project to UTM 10N;
build a point quadtree over the trees (split a cell into 4 while it holds > 150 trees, min
cell 250 m), compute per leaf the number of trees and the percentage with NULL species, clip
the leaves to the city outline and map the percentage (empty leaves left blank).
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box

os.makedirs("pred_results", exist_ok=True)
trees = gpd.read_file("dataset/street_trees_sample.geojson")
sfo = gpd.read_file("dataset/sfo_boroughs.geojson")
trees = trees.assign(species_null=trees["Species"].isnull().astype(int))
print("trees %d, NULL species %d (%.2f %%)" % (len(trees), trees.species_null.sum(), 100 * trees.species_null.mean()))

utm = 32610
t = trees.to_crs(utm)
city = sfo.to_crs(utm)
xy = np.c_[t.geometry.x, t.geometry.y]
nulls = t["species_null"].to_numpy()
NMAX, MIN_SIZE = 150, 250.0


def split(x0, y0, x1, y1, idx, out):
    if len(idx) > NMAX and (x1 - x0) > 2 * MIN_SIZE:
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        px, py = xy[idx, 0], xy[idx, 1]
        for (a, b, c, d), m in (((x0, y0, xm, ym), (px < xm) & (py < ym)), ((xm, y0, x1, ym), (px >= xm) & (py < ym)),
                                ((x0, ym, xm, y1), (px < xm) & (py >= ym)), ((xm, ym, x1, y1), (px >= xm) & (py >= ym))):
            split(a, b, c, d, idx[m], out)
    else:
        out.append((x0, y0, x1, y1, len(idx), int(nulls[idx].sum())))


x0, y0, x1, y1 = city.total_bounds
side = max(x1 - x0, y1 - y0)
leaves = []
split(x0, y0, x0 + side + 1, y0 + side + 1, np.arange(len(xy)), leaves)
quad = gpd.GeoDataFrame(pd.DataFrame(leaves, columns=["x0", "y0", "x1", "y1", "n_trees", "n_null"]),
                        geometry=[box(a, b, c, d) for a, b, c, d, *_ in leaves], crs=utm)
quad["pct_null"] = np.where(quad.n_trees > 0, 100 * quad.n_null / quad.n_trees.clip(lower=1), np.nan)
quad = quad[quad.n_trees > 0]
quad_clipped = gpd.clip(quad, city)
print("%d quadtree leaves with trees; pct_null median %.1f, max %.1f" % (len(quad), quad.pct_null.median(), quad.pct_null.max()))

fig, ax = plt.subplots(figsize=(9, 8))
quad_clipped.plot(ax=ax, column="pct_null", cmap="Greens", scheme="Quantiles", k=5, legend=True,
                  edgecolor="white", linewidth=0.6, legend_kwds={"title": "% trees with NULL species", "loc": "lower left"})
city.boundary.plot(ax=ax, color="0.4", linewidth=1)
ax.set_title("San Francisco street trees: share with missing species (quadtree, <=150 trees per cell)")
ax.set_axis_off()
fig.savefig("pred_results/trees_count_vis.png", dpi=150, bbox_inches="tight")
