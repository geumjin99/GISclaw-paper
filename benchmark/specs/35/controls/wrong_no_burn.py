"""Wrong: the burn-severity layer is left out of the overlay (slope and land cover only)."""
import os
import numpy as np, rasterio, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling
os.makedirs("pred_results", exist_ok=True)
with rasterio.open("dataset/Before_L8.tif") as s:
    pre = s.read().astype("float32"); crs, tr, shape = s.crs, s.transform, (s.height, s.width)
with rasterio.open("dataset/After_L8.tif") as s:
    post = s.read().astype("float32")
NIR, SWIR = 4, 6
def nbr(a):
    with np.errstate(all="ignore"):
        o = (a[NIR] - a[SWIR]) / (a[NIR] + a[SWIR])
    o[(a[NIR] <= 0) | (a[SWIR] <= 0)] = np.nan
    return o
dnbr = nbr(pre) - nbr(post)
dem = np.full(shape, np.nan, "float32")
with rasterio.open("dataset/DEM_30m.tif") as s:
    reproject(s.read(1).astype("float32"), dem, src_transform=s.transform, src_crs=CRS.from_epsg(3857),
              dst_transform=tr, dst_crs=crs, resampling=Resampling.bilinear, src_nodata=0, dst_nodata=np.nan)
gy, gx = np.gradient(dem, 30.0, 30.0)
slope = np.degrees(np.arctan(np.hypot(gx, gy)))
nlcd = np.zeros(shape, "uint8")
with rasterio.open("dataset/Sonoma_NLCD2011.tif") as s:
    reproject(s.read(1), nlcd, src_transform=s.transform, src_crs=s.crs, dst_transform=tr, dst_crs=crs,
              resampling=Resampling.nearest, src_nodata=255, dst_nodata=0)
burn_cls = np.digitize(dnbr, [0.1, 0.27, 0.44, 0.66]) + 1.0
burn_cls[~np.isfinite(dnbr)] = np.nan
slope_cls = np.digitize(slope, [5, 15, 25, 35]) + 1.0
slope_cls[~np.isfinite(slope)] = np.nan
lut = {11: 1, 21: 3, 22: 3, 23: 4, 24: 5, 31: 5, 41: 1, 42: 1, 43: 1, 52: 3, 71: 4, 81: 3, 82: 4, 90: 2, 95: 2}
lc_cls = np.full(shape, np.nan)
for k, v in lut.items():
    lc_cls[nlcd == k] = v
risk = (0.15 * lc_cls + 0.55 * slope_cls) / 0.70
plt.figure(figsize=(9, 9))
plt.imshow(risk, cmap="RdYlGn_r"); plt.colorbar(label="landslide risk"); plt.title("Landslide risk")
plt.savefig("pred_results/landslide_map.png", dpi=120)
