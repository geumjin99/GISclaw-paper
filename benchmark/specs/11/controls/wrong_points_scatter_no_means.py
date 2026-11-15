# Wrong: correct per-point slope/aspect, but the chart is a per-point scatter by
# category; no per-category average is ever taken.
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
fig, ax = plt.subplots(figsize=(10, 7))
for cat, sub in pts.groupby("VernacularNameCategory"):
    ax.scatter(sub["slope"], sub["aspect"], s=4, label=cat)
ax.set_xlabel("Slope (deg)"); ax.set_ylabel("Aspect (deg from N)"); ax.legend(fontsize=6)
plt.tight_layout(); plt.savefig("pred_results/CoralandSponge.png")
