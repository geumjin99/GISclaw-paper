"""Wrong: burn severity from the NDVI difference (B5/B4) instead of the NBR (B5/B7) difference the
task prescribes (dNDVI vs gold dNBR: r 0.92). Otherwise the reference solution.
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
    b5_pre, b7_pre = src.read(5).astype("float32"), src.read(4).astype("float32")  # NIR, red
    grid_crs, grid_tr, shape = src.crs, src.transform, (src.height, src.width)
with rasterio.open("dataset/After_L8.tif") as src:
    b5_post, b7_post = src.read(5).astype("float32"), src.read(4).astype("float32")  # NIR, red


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
