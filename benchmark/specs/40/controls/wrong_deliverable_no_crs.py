"""Wrong: the reference analysis (change in malaria prevalence, 2015 minus 2000), but every
geographic deliverable file is then rewritten without a CRS.

Minus: change = PR2015 - PR2000 (negative = decrease), nodata (-9999, ocean / outside
Africa) masked; classified with the original authors' 5-class remap; mapped with a
diverging colour scheme centred on zero.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm

os.makedirs("pred_results", exist_ok=True)

with rasterio.open("dataset/2015_Nature_Africa_PR.2000.tif") as src:
    pr2000 = src.read(1).astype("float32")
    profile = src.profile.copy()
    nd = src.nodata
with rasterio.open("dataset/2015_Nature_Africa_PR.2015.tif") as src:
    pr2015 = src.read(1).astype("float32")

valid = (pr2000 != nd) & (pr2015 != nd) & np.isfinite(pr2000) & np.isfinite(pr2015)
change = np.where(valid, pr2015 - pr2000, np.nan).astype("float32")

# original authors' remap (RemapRange): 1 significant decrease ... 5 significant increase
classes = np.zeros(change.shape, dtype="uint8")
edges = [(-1.0, -0.3, 1), (-0.3, -0.1, 2), (-0.1, 0.1, 3), (0.1, 0.3, 4), (0.3, 1.0001, 5)]
for lo, hi, c in edges:
    classes[valid & (change >= lo) & (change < hi)] = c

profile.update(dtype="float32", count=1, nodata=-9999, compress="lzw")
with rasterio.open("pred_results/Malaria_Change.tif", "w", **profile) as dst:
    dst.write(np.where(valid, change, -9999).astype("float32"), 1)
profile.update(dtype="uint8", nodata=0)
with rasterio.open("pred_results/Malaria_Change_Classified.tif", "w", **profile) as dst:
    dst.write(classes, 1)

b = rasterio.open("dataset/2015_Nature_Africa_PR.2000.tif").bounds
ext = [b.left, b.right, b.bottom, b.top]
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
lim = float(np.nanmax(np.abs(change)))
im = axes[0].imshow(change, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), extent=ext)
axes[0].set_title("Change in PfPR(2-10), 2015 minus 2000")
fig.colorbar(im, ax=axes[0], shrink=0.7, label="change in prevalence (red = increase)")
cmap5 = ListedColormap(["#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#b2182b"])
cl = np.where(classes == 0, np.nan, classes.astype(float))
im2 = axes[1].imshow(cl, cmap=cmap5, norm=BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], 5), extent=ext)
cb = fig.colorbar(im2, ax=axes[1], shrink=0.7, ticks=[1, 2, 3, 4, 5])
cb.ax.set_yticklabels(["large decrease (<-0.3)", "decrease (-0.3..-0.1)", "little change",
                       "increase (0.1..0.3)", "large increase (>0.3)"])
axes[1].set_title("Classified change")
for a in axes:
    a.set_xlabel("Longitude"); a.set_ylabel("Latitude")
plt.tight_layout()
plt.savefig("pred_results/malaria_change.png", dpi=150)
print("mean change", float(np.nanmean(change)))


# ---------------------------------------------------------------------------------------------
# Negative control: the analysis above is the reference solution, but every geographic
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


_strip_deliverable_crs(['*.tif*'])
