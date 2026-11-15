"""Reference solution for task 45: terrain ruggedness and ruggedness cost for mountain-lion corridors.

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
