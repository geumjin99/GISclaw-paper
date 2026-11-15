"""Reference solution for task 24: spatial de-identification of child blood-lead points by coordinate rounding.

Adapted from the reference script of GeoAnalystBench (Zhang et al. 2025, Apache-2.0).
"""
import geopandas as gpd
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Point


def main():
    points = gpd.read_file("dataset/High_Blood_Level_Results.shp")   # EPSG:3857 (Web Mercator)
    zips = gpd.read_file("dataset/Sacramento_ZIP_Codes.shp")         # EPSG:4326
    # Consistent CRS: WGS 1984, so that "two decimals" means 0.01 degree (~1.1 km x 0.9 km)
    points = points.to_crs("EPSG:4326")
    zips = zips.to_crs("EPSG:4326")

    points["Longitude"] = points.geometry.x
    points["Latitude"] = points.geometry.y
    points["LonRound"] = points["Longitude"].round(2)
    points["LatRound"] = points["Latitude"].round(2)

    rounded = points.drop(columns="geometry").copy()
    rounded = gpd.GeoDataFrame(rounded, geometry=[Point(xy) for xy in zip(rounded.LonRound, rounded.LatRound)],
                               crs="EPSG:4326")
    # de-identified layer keeps only the generalised location (exact coordinates removed)
    rounded = rounded.drop(columns=["Longitude", "Latitude"])

    n_records = len(points)
    n_unique_before = points[["Longitude", "Latitude"]].round(8).drop_duplicates().shape[0]
    n_unique_after = rounded[["LonRound", "LatRound"]].drop_duplicates().shape[0]
    reduction = n_unique_before - n_unique_after
    reduction_pct = 100.0 * reduction / n_unique_before
    print(f"records {n_records}; unique locations {n_unique_before} -> {n_unique_after} "
          f"(reduction {reduction}, {reduction_pct:.1f} %)")
    summary = pd.DataFrame({"metric": ["records", "unique_locations_original", "unique_locations_rounded",
                                       "reduction", "reduction_pct"],
                            "value": [n_records, n_unique_before, n_unique_after, reduction, round(reduction_pct, 2)]})
    summary.to_csv("pred_results/rounding_summary.csv", index=False)

    rounded.to_file("pred_results/High_Blood_Level_Results_Rounded.shp")

    counts = rounded.groupby(["LonRound", "LatRound"]).size().rename("n").reset_index()
    fig, ax = plt.subplots(figsize=(8, 8))
    zips.boundary.plot(ax=ax, color="grey", linewidth=0.6)
    points.plot(ax=ax, color="lightcoral", markersize=4, label="original")
    ax.scatter(counts.LonRound, counts.LatRound, s=counts.n * 8, facecolors="none", edgecolors="navy",
               label="rounded (size = cases)")
    ax.legend()
    ax.set_title(f"Coordinate rounding (2 decimals): {n_unique_before} -> {n_unique_after} unique locations")
    fig.savefig("pred_results/rounded_points.png", dpi=110, bbox_inches="tight")


main()
