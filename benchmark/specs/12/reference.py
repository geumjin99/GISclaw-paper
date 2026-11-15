"""Reference solution for task 12: Terrain Ruggedness Index (TRI) from the elevation raster (mountain lion
habitat).

TRI as defined in the task: for every cell, the elevation differences to its 8 neighbours are
squared, averaged and square-rooted (root-mean-square neighbour difference). Cells outside the
data (alpha band = 0) are NoData; at NoData/edge neighbours the mean is taken over the valid
neighbours only.

Note: Elevation.tif is an 8-bit RGBA export (bands 1-3 identical grey values 0-255, band 4 =
alpha), not a DEM in metres, so TRI is in grey-level units (relative ruggedness).
"""
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

os.makedirs("pred_results", exist_ok=True)

with rasterio.open("dataset/Elevation.tif") as src:
    elevation = src.read(1).astype("float64")
    valid = src.read(4) == 255 if src.count == 4 else np.ones(elevation.shape, bool)
    extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
elevation[~valid] = np.nan

H, W = elevation.shape
padded = np.pad(elevation, 1, constant_values=np.nan)
sq_sum = np.zeros((H, W))
n_valid = np.zeros((H, W))
for dy in (-1, 0, 1):
    for dx in (-1, 0, 1):
        if dy == 0 and dx == 0:
            continue
        nb = padded[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        d2 = (nb - elevation) ** 2
        ok = np.isfinite(d2)
        sq_sum[ok] += d2[ok]
        n_valid += ok
with np.errstate(invalid="ignore", divide="ignore"):
    ruggedness = np.sqrt(sq_sum / n_valid)
ruggedness[~valid | (n_valid == 0)] = np.nan
print("TRI stats: mean %.3f  p50 %.3f  p95 %.3f  max %.3f" % (
    np.nanmean(ruggedness), np.nanpercentile(ruggedness, 50), np.nanpercentile(ruggedness, 95), np.nanmax(ruggedness)))

fig, ax = plt.subplots(figsize=(11, 8))
im = ax.imshow(ruggedness, cmap="magma", extent=extent, vmin=0, vmax=np.nanpercentile(ruggedness, 99))
fig.colorbar(im, ax=ax, label="TRI (RMS elevation difference to 8 neighbours, grey-level units)")
ax.set_title("Terrain Ruggedness Index")
ax.set_xlabel("Easting (m, NAD83 / UTM 11N)")
ax.set_ylabel("Northing (m)")
fig.savefig("pred_results/ruggedness.png", dpi=150, bbox_inches="tight")
