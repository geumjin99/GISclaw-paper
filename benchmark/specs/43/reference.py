"""Reference solution for task 43: elk home range - convex hull, kernel density, density-based
clustering.

Mirrors the original ArcPy model on the 150 191 GPS fixes of 2009:
  1. Minimum Bounding Geometry, CONVEX_HULL, no grouping -> one home-range polygon;
  2. Kernel Density (planar, quartic kernel as in ArcGIS, search radius 468.8 m,
     densities per km2) computed in NAD83 / Alberta 10-TM (Forest) = EPSG:3400, the output
     coordinate system named in the original model;
  3. Density-based clustering with HDBSCAN, minimum cluster size 100.
Outputs (pred_results/): elk_home_range.geojson, elk_kernel_density.tif, elk_clusters.geojson,
elk_analysis.png.
"""
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from scipy.signal import fftconvolve
from sklearn.cluster import HDBSCAN
import matplotlib.pyplot as plt

CRS = "EPSG:3400"            # NAD83 / Alberta 10-TM (Forest)
RADIUS = 468.805393200003    # search radius of the original model (m)
CELL = 100.0                 # output cell size (m)

elk = gpd.read_file("dataset/Elk_in_Southwestern_Alberta_2009.geojson").to_crs(CRS)
xy = np.c_[elk.geometry.x.values, elk.geometry.y.values]

# 1. Minimum bounding geometry - convex hull of all fixes
home_range = gpd.GeoDataFrame({"name": ["elk_2009_convex_hull"]},
                              geometry=[elk.geometry.union_all().convex_hull], crs=CRS)
home_range["area_km2"] = home_range.area / 1e6
home_range.to_file("pred_results/elk_home_range.geojson", driver="GeoJSON")
print("home range (convex hull) km2:", round(float(home_range.area_km2.iloc[0]), 1))

# 2. Kernel density (quartic kernel, planar) on a 100 m grid
xmin, ymin = xy.min(0) - RADIUS
xmax, ymax = xy.max(0) + RADIUS
ncol = int(np.ceil((xmax - xmin) / CELL)); nrow = int(np.ceil((ymax - ymin) / CELL))
col = ((xy[:, 0] - xmin) / CELL).astype(int); row = ((ymax - xy[:, 1]) / CELL).astype(int)
counts = np.zeros((nrow, ncol)); np.add.at(counts, (row, col), 1)
k = int(np.ceil(RADIUS / CELL))
dy, dx = np.mgrid[-k:k + 1, -k:k + 1] * CELL
d2 = (dx ** 2 + dy ** 2) / RADIUS ** 2
kern = np.where(d2 < 1, 3.0 / (np.pi * RADIUS ** 2) * (1 - d2) ** 2, 0.0)   # per m2, integrates to 1
density = fftconvolve(counts, kern, mode="same") * 1e6                     # points per km2
density[density < 1e-9] = 0.0
kd_transform = from_origin(xmin, ymax, CELL, CELL)
with rasterio.open("pred_results/elk_kernel_density.tif", "w", driver="GTiff", height=nrow, width=ncol,
                   count=1, dtype="float32", crs=CRS, transform=kd_transform, compress="lzw") as dst:
    dst.write(density.astype("float32"), 1)
print("KD grid", density.shape, "max per km2", round(float(density.max()), 1))

# 3. Density-based clustering (HDBSCAN, min cluster size 100)
elk["cluster"] = HDBSCAN(min_cluster_size=100).fit_predict(xy)
n_clusters = int(elk["cluster"].max() + 1)
print("clusters:", n_clusters, "noise share:", round(float((elk.cluster == -1).mean()), 3))
elk[["OBJECTID", "timestamp", "ind_ident", "cluster", "geometry"]].to_file(
    "pred_results/elk_clusters.geojson", driver="GeoJSON")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
elk.plot(ax=axes[0], markersize=0.2, color="grey"); home_range.boundary.plot(ax=axes[0], color="red")
axes[0].set_title("Home range (convex hull)")
axes[1].imshow(np.log1p(density), extent=[xmin, xmin + ncol * CELL, ymax - nrow * CELL, ymax], cmap="magma")
axes[1].set_title("Kernel density (log, fixes/km2)")
elk[elk.cluster >= 0].plot(ax=axes[2], column="cluster", cmap="tab20", markersize=0.2)
axes[2].set_title(f"HDBSCAN clusters (n={n_clusters})")
plt.tight_layout()
plt.savefig("pred_results/elk_analysis.png", dpi=120)
