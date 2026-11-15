"""Wrong: the analysis is the reference solution, but every geographic deliverable file is rewritten
without a CRS; the deliverable contract requires a CRS.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling

os.makedirs("pred_results", exist_ok=True)

with rasterio.open("dataset/Before_L8.tif") as src:
    b5_pre, b7_pre = src.read(5).astype("float32"), src.read(7).astype("float32")
    grid_crs, grid_tr, shape = src.crs, src.transform, (src.height, src.width)
with rasterio.open("dataset/After_L8.tif") as src:
    b5_post, b7_post = src.read(5).astype("float32"), src.read(7).astype("float32")


def nbr(b5, b7):
    den = b5 + b7
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (b5 - b7) / den
    out[(den <= 0) | (b5 <= 0) | (b7 <= 0)] = np.nan  # fill / invalid SR
    return out


nbr_pre, nbr_post = nbr(b5_pre, b7_pre), nbr(b5_post, b7_post)
dnbr = nbr_pre - nbr_post

# DEM -> Landsat grid
dem = np.full(shape, np.nan, dtype="float32")
with rasterio.open("dataset/DEM_30m.tif") as src:
    reproject(src.read(1).astype("float32"), dem, src_transform=src.transform, src_crs=CRS.from_epsg(3857),
              dst_transform=grid_tr, dst_crs=grid_crs, resampling=Resampling.bilinear, src_nodata=0,
              dst_nodata=np.nan)
dy, dx = np.gradient(dem, grid_tr.e * -1, grid_tr.a)  # metres
slope = np.degrees(np.arctan(np.hypot(dx, dy)))

# NLCD -> Landsat grid
nlcd = np.zeros(shape, dtype="uint8")
with rasterio.open("dataset/Sonoma_NLCD2011.tif") as src:
    reproject(src.read(1), nlcd, src_transform=src.transform, src_crs=src.crs, dst_transform=grid_tr,
              dst_crs=grid_crs, resampling=Resampling.nearest, src_nodata=255, dst_nodata=0)


def remap(a, breaks):
    out = np.zeros(a.shape, dtype="float32")
    for lo, hi, v in breaks:
        out[(a >= lo) & (a < hi)] = v
    out[~np.isfinite(a)] = 0
    return out


burn_class = remap(dnbr, [(-np.inf, 0.10, 1), (0.10, 0.27, 2), (0.27, 0.44, 3), (0.44, 0.66, 4), (0.66, np.inf, 5)])
slope_class = remap(slope, [(0, 5, 1), (5, 15, 2), (15, 25, 3), (25, 35, 4), (35, 91, 5)])
lc = nlcd.astype("float32")
lc_class = remap(lc, [(1, 12.5, 1), (21, 23, 3), (23, 24, 4), (24, 32, 5), (41, 45, 1), (52, 53, 3),
                      (71, 72, 4), (81, 82, 3), (82, 83, 4), (90, 96, 2)])

valid = (burn_class > 0) & (slope_class > 0) & (lc_class > 0)
landslide_risk = np.where(valid, 0.15 * lc_class + 0.30 * burn_class + 0.55 * slope_class, np.nan).astype("float32")
print("valid cells:", int(valid.sum()), "risk mean:", float(np.nanmean(landslide_risk)))

profile = dict(driver="GTiff", height=shape[0], width=shape[1], count=1, dtype="float32", crs=grid_crs,
               transform=grid_tr, nodata=np.nan, compress="lzw")
with rasterio.open("pred_results/landslide_risk.tif", "w", **profile) as dst:
    dst.write(landslide_risk, 1)

fig, axes = plt.subplots(2, 2, figsize=(14, 14))
panels = [(burn_class, "Burn severity class (dNBR)"), (slope_class, "Slope class"), (lc_class, "Land cover class"),
          (landslide_risk, "Landslide risk (weighted overlay 30/55/15)")]
for ax, (a, t) in zip(axes.flat, panels):
    im = ax.imshow(np.where(a > 0, a, np.nan), cmap="RdYlGn_r", vmin=1, vmax=5)
    ax.set_title(t); ax.set_axis_off()
    fig.colorbar(im, ax=ax, shrink=0.7)
plt.tight_layout()
plt.savefig("pred_results/landslide_map.png", dpi=150)


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


_strip_deliverable_crs(['*.tif*'])
