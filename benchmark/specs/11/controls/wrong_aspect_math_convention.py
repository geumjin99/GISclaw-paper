# Wrong: aspect from arctan2(dz/dy, dz/dx) - mathematical angle (0 = east, counter-clockwise), not compass.
import numpy as np, pandas as pd, geopandas as gpd, rasterio, matplotlib.pyplot as plt
with rasterio.open("dataset/CatalinaBathymetry.tif") as src:
    z = src.read(1).astype(float); T = src.transform
pts = gpd.read_file("dataset/CoralandSpongeCatalina.geojson")
p = pts.to_crs(3310)
r, c = rasterio.transform.rowcol(T, p.geometry.x.values, p.geometry.y.values)
r, c = np.clip(np.asarray(r), 0, z.shape[0] - 1), np.clip(np.asarray(c), 0, z.shape[1] - 1)

dzdy, dzdx = np.gradient(z, -T.a, T.a)
slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
aspect = np.degrees(np.arctan2(dzdy, dzdx)) % 360
pts["slope"] = slope[r, c]; pts["aspect"] = aspect[r, c]
stats = pts.groupby("VernacularNameCategory")[["slope", "aspect"]].mean()
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
stats["slope"].plot.bar(ax=ax[0]); stats["aspect"].plot.bar(ax=ax[1])
plt.tight_layout(); plt.savefig("pred_results/CoralandSponge.png")
