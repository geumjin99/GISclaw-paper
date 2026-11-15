# Correct: per-species means via pivot_table(aggfunc='mean') inside the
# plotting call, grouped by ScientificName.
import numpy as np, pandas as pd, geopandas as gpd, rasterio, matplotlib.pyplot as plt
with rasterio.open("dataset/CatalinaBathymetry.tif") as src:
    z = src.read(1).astype(float); T = src.transform
pts = gpd.read_file("dataset/CoralandSpongeCatalina.geojson")
p = pts.to_crs(3310)
r, c = rasterio.transform.rowcol(T, p.geometry.x.values, p.geometry.y.values)
r, c = np.clip(np.asarray(r), 0, z.shape[0] - 1), np.clip(np.asarray(c), 0, z.shape[1] - 1)

from scipy.ndimage import convolve
kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / (8 * T.a)
ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / (8 * T.a)
dzdx = convolve(z, kx[:, ::-1][::-1], mode="nearest")   # east gradient
dzdy = convolve(z, ky[:, ::-1][::-1], mode="nearest")   # south gradient
pts["slope"] = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))[r, c]
pts["aspect"] = ((np.degrees(np.arctan2(dzdx, -dzdy)) + 180) % 360)[r, c]
pts.pivot_table(index="ScientificName", values=["slope", "aspect"], aggfunc="mean").plot.barh(
    subplots=True, layout=(1, 2), figsize=(14, 12), legend=False)
plt.tight_layout(); plt.savefig("pred_results/CoralandSponge.png")
