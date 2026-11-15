"""Wrong: the reference analysis of task 24 (de-identification by coordinate rounding), with every geographic
deliverable then written without a CRS; the deliverables must be georeferenced.

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


# ---------------------------------------------------------------------------------------------
# NEGATIVE CONTROL: the analysis above is the reference solution, but every geographic
# deliverable file is then written without a CRS. Expected: wrong.
def _write_without_crs(path):
    """Rewrite one file in place so that it carries no CRS. Returns a short description.
    .shp: drop the .prj sidecar. .tif/.tiff: rewrite without CRS (and drop .aux.xml, which can
    hold one). .gpkg: rewrite every spatial layer with crs=None. .geojson/.json: write projected
    coordinates (UTM if the data are lon/lat) without a 'crs' member - a lon/lat GeoJSON
    without 'crs' is WGS 84 by definition (RFC 7946), so dropping the member alone would not
    remove the CRS."""
    import os
    low = path.lower()
    stem, _ = os.path.splitext(path)
    if low.endswith(".shp"):
        d, b = os.path.split(stem)
        gone = []
        for n in os.listdir(d or "."):
            if n.lower() == (b + ".prj").lower():
                os.remove(os.path.join(d, n))
                gone.append(n)
        return f"removed {gone}"
    if low.endswith((".tif", ".tiff")):
        import rasterio
        with rasterio.open(path) as r:
            prof, data = r.profile.copy(), r.read()
            tags = r.tags()
        prof.pop("crs", None)
        tmp = stem + ".__nocrs__" + os.path.splitext(path)[1]
        with rasterio.open(tmp, "w", **prof) as w:
            w.write(data)
            if tags:
                w.update_tags(**tags)
        os.replace(tmp, path)
        if os.path.exists(path + ".aux.xml"):
            os.remove(path + ".aux.xml")
        return "raster rewritten without CRS"
    if low.endswith(".gpkg"):
        import geopandas as gpd
        import pyogrio
        tmp = stem + ".__nocrs__.gpkg"
        for lyr, gt in pyogrio.list_layers(path):
            g = gpd.read_file(path, layer=lyr)
            if gt is not None:
                g = g.set_crs(None, allow_override=True)
            g.to_file(tmp, layer=lyr, driver="GPKG")
        os.replace(tmp, path)
        return "gpkg rewritten without CRS"
    if low.endswith((".geojson", ".json")):
        import geopandas as gpd
        g = gpd.read_file(path)
        if g.crs is None or g.crs.is_geographic:
            if g.crs is None:
                g = g.set_crs(4326)
            g = g.to_crs(g.estimate_utm_crs())
        g = g.set_crs(None, allow_override=True)
        os.remove(path)
        g.to_file(path, driver="GeoJSON")
        return "GeoJSON rewritten in projected coordinates without 'crs'"
    return "not a geographic file"


def _strip_deliverable_crs(patterns):
    import fnmatch
    import os
    for n in sorted(os.listdir("pred_results")):
        low = n.lower()
        if low.endswith(('.shp', '.geojson', '.gpkg', '.json', '.tif', '.tiff')) and any(fnmatch.fnmatchcase(low, p.lower()) for p in patterns):
            if low.endswith(".json"):
                try:
                    import json
                    if json.load(open(os.path.join("pred_results", n))).get("type") is None:
                        continue
                except Exception:
                    continue
            print("no-CRS control:", n, _write_without_crs(os.path.join("pred_results", n)))


_strip_deliverable_crs(['*.geojson', '*.gpkg', '*.json', '*.shp'])
