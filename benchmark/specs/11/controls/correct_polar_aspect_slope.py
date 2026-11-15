# Correct: polar chart - each category drawn at its circular mean aspect
# (angle, clockwise from north) with its mean slope as radius.
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
fig = plt.figure(figsize=(9, 9)); ax = fig.add_subplot(projection="polar")
ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
ax.bar(np.radians(summary.mean_aspect), summary.mean_slope, width=0.08, alpha=0.7)
for _, row in summary.iterrows():
    ax.annotate(row.VernacularNameCategory, (np.radians(row.mean_aspect), row.mean_slope), fontsize=7)
ax.set_title("Mean aspect (angle) and mean slope (radius, deg) per category")
plt.savefig("pred_results/CoralandSponge.png")
