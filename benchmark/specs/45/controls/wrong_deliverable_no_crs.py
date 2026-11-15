"""Wrong: the reference solution of task 45 with every geographic deliverable written without a CRS.

The analysis: terrain ruggedness and ruggedness cost for mountain-lion corridors.

Elevation.tif (30 m, NAD83 / UTM 11N, metres; Learn ArcGIS mountain-lion project).
  1. Ruggedness = Focal Statistics, 3x3 rectangle, RANGE (max - min), NoData ignored ("DATA"),
     exactly as in the benchmark's ArcPy model;
  2. Ruggedness_Cost = Rescale by Function, MSLarge (lower threshold 0, upper threshold =
     ruggedness maximum, mean multiplier 1, std multiplier 1), evaluation scale from 10 to 1:
     flat terrain -> cost 10, very rugged terrain (preferred by mountain lions) -> cost ~1.
     MSLarge: f(x) = 1 - b*s / (x - a*m + b*s) for x > a*m, else 0; out = 10 - 9 f.
Outputs: pred_results/Ruggedness.tif, pred_results/Ruggedness_Cost.tif, pred_results/ruggedness_cost.png
"""
import numpy as np
import rasterio
from scipy.ndimage import maximum_filter, minimum_filter
import matplotlib.pyplot as plt

with rasterio.open("dataset/Elevation.tif") as src:
    elev = src.read(1).astype("float64")
    profile = src.profile.copy()
    nodata = src.nodata
valid = np.isfinite(elev) & (elev != nodata) & (elev > -1e30) & (elev < 1e30)
elev[~valid] = np.nan
print("DEM", elev.shape, "valid share", round(float(valid.mean()), 4), "range", np.nanmin(elev), np.nanmax(elev))

# 1. focal statistics: 3x3 RANGE, ignoring NoData neighbours
hi = maximum_filter(np.where(valid, elev, -np.inf), size=3, mode="nearest")
lo = minimum_filter(np.where(valid, elev, np.inf), size=3, mode="nearest")
ruggedness = np.where(valid, hi - lo, np.nan)
print("ruggedness: mean %.2f std %.2f max %.2f" % (np.nanmean(ruggedness), np.nanstd(ruggedness), np.nanmax(ruggedness)))

# 2. rescale by function: MSLarge, scale 10 -> 1
m, s = np.nanmean(ruggedness), np.nanstd(ruggedness)
a_mult, b_mult = 1.0, 1.0
x = ruggedness
f = np.where(x > a_mult * m, 1.0 - (b_mult * s) / (x - a_mult * m + b_mult * s), 0.0)
ruggedness_cost = np.where(valid, 10.0 - 9.0 * f, np.nan)
print("cost range", np.nanmin(ruggedness_cost), np.nanmax(ruggedness_cost))

profile.update(dtype="float32", nodata=-9999.0, count=1, compress="lzw")
for name, arr in (("Ruggedness", ruggedness), ("Ruggedness_Cost", ruggedness_cost)):
    with rasterio.open(f"pred_results/{name}.tif", "w", **profile) as dst:
        dst.write(np.where(np.isfinite(arr), arr, -9999.0).astype("float32"), 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
im0 = axes[0].imshow(ruggedness, cmap="viridis", vmax=np.nanpercentile(ruggedness, 99)); axes[0].set_title("Ruggedness (3x3 range, m)")
plt.colorbar(im0, ax=axes[0], shrink=0.7)
im1 = axes[1].imshow(ruggedness_cost, cmap="RdYlGn_r"); axes[1].set_title("Ruggedness cost (MSLarge, 10 = flat -> 1 = rugged)")
plt.colorbar(im1, ax=axes[1], shrink=0.7)
plt.tight_layout()
plt.savefig("pred_results/ruggedness_cost.png", dpi=100)


# ---------------------------------------------------------------------------------------------
# Negative control: the analysis above is the reference solution, but every geographic
# deliverable file is then rewritten without a CRS, which violates the deliverable contract.
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


_strip_deliverable_crs(['*.asc', '*.img', '*.tif', '*.tiff'])
