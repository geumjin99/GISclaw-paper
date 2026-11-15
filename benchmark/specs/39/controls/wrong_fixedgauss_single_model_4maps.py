"""Wrong: fixed gaussian kernel on polygon centroids with standardised X (a legitimate model), but the
figure shows only that one model's intercept and three slopes; no models are compared.
"""
import geopandas as gpd, numpy as np, matplotlib.pyplot as plt
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import GWR
g = gpd.read_file("dataset/G_utm.shp")
cent = g.geometry.centroid
coords = np.c_[cent.x, cent.y]
y = g.PctBach.values.reshape(-1, 1)
Xr = g[["PctFB", "PctBlack", "PctRural"]].values
Xs = (Xr - Xr.mean(0)) / Xr.std(0)
best_bw = Sel_BW(coords, y, Xs, fixed=True, kernel="gaussian").search()
res = GWR(coords, y, Xs, bw=best_bw, fixed=True, kernel="gaussian").fit()
fig, ax = plt.subplots(2, 2, figsize=(12, 11))
for a, j, n in zip(ax.flat, range(4), ["Intercept", "PctFB", "PctBlack", "PctRural"]):
    g.assign(b=res.params[:, j]).plot("b", ax=a, cmap="RdBu_r", legend=True)
    a.set_title(f"{n} (fixed gaussian bw {best_bw/1000:.0f} km)"); a.set_axis_off()
plt.savefig("pred_results/GA_GWR.png")
