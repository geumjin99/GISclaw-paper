# Correct: slope and aspect means as grouped vertical bars in one axes.
import numpy as np, pandas as pd, geopandas as gpd, rasterio, matplotlib.pyplot as plt
with rasterio.open("dataset/CatalinaBathymetry.tif") as src:
    z = src.read(1).astype(float); T = src.transform
pts = gpd.read_file("dataset/CoralandSpongeCatalina.geojson")
p = pts.to_crs(3310)
r, c = rasterio.transform.rowcol(T, p.geometry.x.values, p.geometry.y.values)
r, c = np.clip(np.asarray(r), 0, z.shape[0] - 1), np.clip(np.asarray(c), 0, z.shape[1] - 1)
dzdy, dzdx = np.gradient(z, T.a, T.a)                      # dzdy along rows (southwards)
pts["slope"] = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))[r, c]
pts["aspect"] = (np.degrees(np.arctan2(-dzdx, dzdy)) % 360)[r, c]   # downslope azimuth
def circ(a):
    a = np.radians(a); return np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360
summary = pts.groupby("VernacularNameCategory").agg(mean_slope=("slope", "mean"), mean_aspect=("aspect", circ))
summary = summary.reset_index()
ax = summary.set_index("VernacularNameCategory")[["mean_slope", "mean_aspect"]].plot.bar(figsize=(12, 6), logy=True)
ax.set_ylabel("deg (log)"); plt.tight_layout(); plt.savefig("pred_results/CoralandSponge.png")
