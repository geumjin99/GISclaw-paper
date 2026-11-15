"""Reference solution for task 40: change in malaria prevalence (PfPR 2-10), Sub-Saharan
Africa, 2000 -> 2015.

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
