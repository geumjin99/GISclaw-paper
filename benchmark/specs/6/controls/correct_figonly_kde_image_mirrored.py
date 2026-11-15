"""Correct: the analysis is right (hulls, KDE, DBSCAN as the reference); only the presentation is
wrong - the density image is drawn mirrored north-south (origin='lower' with extent bottom/top
swapped). The required checks pass; the optional figure check fails.

Steps, as in the reference solution:

  * project the 150 191 GPS fixes (EPSG:4326) to UTM 11N (EPSG:32611, metres) - distances,
    kernel bandwidths and DBSCAN radii must be metric (a 468.8 m bandwidth applied to
    degrees flattens the density surface completely);
  * Minimum Bounding Geometry: convex hull of all fixes (population home range) and one hull
    per animal (ind_ident) for the per-animal ranges;
  * Kernel density: Gaussian kernel, bandwidth 468.8 m (ArcGIS default of the benchmark's
    workflow), on a 200 m grid over the fix extent (histogram + Gaussian filter = exact binned KDE);
  * DBSCAN (sklearn) with eps = 500 m and min_samples = 100 fixes;
  * overlay: density surface, population hull, per-animal hulls and DBSCAN clusters on one map.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

CRS_M = "EPSG:32611"
BW = 468.805393200003  # m
CELL = 200.0


def main():
    elk = gpd.read_file("dataset/Elk_in_Southwestern_Alberta_2009.geojson").to_crs(CRS_M)
    xy = np.c_[elk.geometry.x, elk.geometry.y]

    # Minimum bounding geometry (convex hull)
    home_range = gpd.GeoDataFrame({"name": ["all elk"]}, geometry=[elk.geometry.union_all().convex_hull], crs=CRS_M)
    hulls = elk.dissolve(by="ind_ident").convex_hull
    animal_ranges = gpd.GeoDataFrame({"ind_ident": hulls.index, "area_km2": hulls.area.values / 1e6},
                                     geometry=hulls.values, crs=CRS_M)
    print(f"population home range {home_range.area.iloc[0] / 1e6:.0f} km2; "
          f"{len(animal_ranges)} animals, median range {animal_ranges.area_km2.median():.0f} km2")

    # Kernel density (binned Gaussian KDE, density in fixes per km2)
    xmin, ymin, xmax, ymax = elk.total_bounds
    xe = np.arange(xmin - 5 * BW, xmax + 5 * BW + CELL, CELL)
    ye = np.arange(ymin - 5 * BW, ymax + 5 * BW + CELL, CELL)
    H, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[xe, ye])
    kde = gaussian_filter(H.T, sigma=BW / CELL, mode="constant") / (CELL * CELL / 1e6)
    kde = kde[::-1]  # north up
    kde_extent = (xe[0], xe[-1], ye[0], ye[-1])

    # DBSCAN
    db = DBSCAN(eps=500, min_samples=100).fit(xy)
    elk["cluster"] = db.labels_
    n_cl = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    print(f"DBSCAN: {n_cl} clusters, noise {np.mean(db.labels_ == -1):.1%}")
    cluster_sizes = elk[elk.cluster >= 0].groupby("cluster").size().sort_values(ascending=False)
    print("largest clusters:", cluster_sizes.head(8).to_dict())

    fig, ax = plt.subplots(figsize=(12, 12))
    show = np.where(kde > 0.05 * kde.max() / 100, kde, np.nan)
    south_first = np.sqrt(show)[::-1]
    im = ax.imshow(south_first, origin="lower",
                   extent=(kde_extent[0], kde_extent[1], kde_extent[3], kde_extent[2]),  # bottom/top swapped
                   cmap="YlOrBr", alpha=0.85, zorder=1)
    ax.set_ylim(kde_extent[2], kde_extent[3])
    cb = plt.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("Kernel density (sqrt of fixes per km$^2$)")
    animal_ranges.boundary.plot(ax=ax, color="grey", linewidth=0.4, zorder=2)
    home_range.boundary.plot(ax=ax, color="red", linewidth=2, zorder=3)
    cl = elk[elk.cluster >= 0]
    sample = cl.sample(min(len(cl), 30000), random_state=0)
    ax.scatter(sample.geometry.x, sample.geometry.y, c=sample.cluster, cmap="tab20", s=0.5, zorder=4)
    ax.set_title(f"Elk 2009: convex-hull home range (red), per-animal hulls (grey),\n"
                 f"kernel density (bandwidth {BW:.0f} m) and {n_cl} DBSCAN clusters (eps 500 m, min 100 fixes)")
    ax.set_xlabel("Easting (m, UTM 11N)")
    ax.set_ylabel("Northing (m)")
    plt.savefig("pred_results/Elk_Analysis.png", dpi=150, bbox_inches="tight")
    return elk, home_range, animal_ranges, kde


elk, home_range, animal_ranges, kde = main()
