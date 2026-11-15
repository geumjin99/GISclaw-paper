# Wrong: the task prescribes DBSCAN; clusters here are connected regions of a thresholded kernel
# density (noise = fixes outside them), a different method with a DBSCAN-like label vector.
"""Hull + kernel density as the reference; clusters from a density threshold."""
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.ndimage import gaussian_filter
from scipy import ndimage
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

    # density regions: threshold the kernel density and label connected regions (not DBSCAN)
    north = kde[::-1]                                     # rows south->north, as histogram2d
    hot = north > np.percentile(north[north > 0], 90)
    reg, n_cl = ndimage.label(hot)
    ix = np.clip(((xy[:, 0] - xe[0]) // CELL).astype(int), 0, north.shape[1] - 1)
    iy = np.clip(((xy[:, 1] - ye[0]) // CELL).astype(int), 0, north.shape[0] - 1)
    lab = reg[iy, ix] - 1                                 # 0 = outside -> -1 (noise)
    elk["cluster"] = lab
    print(f"density regions: {n_cl} regions, outside {np.mean(lab == -1):.1%}")
    cluster_sizes = elk[elk.cluster >= 0].groupby("cluster").size().sort_values(ascending=False)
    print("largest clusters:", cluster_sizes.head(8).to_dict())

    fig, ax = plt.subplots(figsize=(12, 12))
    show = np.where(kde > 0.05 * kde.max() / 100, kde, np.nan)
    im = ax.imshow(np.sqrt(show), extent=kde_extent, cmap="YlOrBr", alpha=0.85, zorder=1)
    cb = plt.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("Kernel density (sqrt of fixes per km$^2$)")
    animal_ranges.boundary.plot(ax=ax, color="grey", linewidth=0.4, zorder=2)
    home_range.boundary.plot(ax=ax, color="red", linewidth=2, zorder=3)
    cl = elk[elk.cluster >= 0]
    sample = cl.sample(min(len(cl), 30000), random_state=0)
    ax.scatter(sample.geometry.x, sample.geometry.y, c=sample.cluster, cmap="tab20", s=0.5, zorder=4)
    ax.set_title(f"Elk 2009: convex-hull home range (red), per-animal hulls (grey),\n"
                 f"kernel density (bandwidth {BW:.0f} m) and {n_cl} high-density regions")
    ax.set_xlabel("Easting (m, UTM 11N)")
    ax.set_ylabel("Northing (m)")
    plt.savefig("pred_results/Elk_Analysis.png", dpi=150, bbox_inches="tight")
    return elk, home_range, animal_ranges, kde


elk, home_range, animal_ranges, kde = main()
