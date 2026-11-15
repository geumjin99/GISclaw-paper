"""Correct: fixed gaussian kernel, coordinates = polygon centroids, standardised X; AICc profile and
coefficient maps.
"""
import geopandas as gpd, numpy as np, pandas as pd, matplotlib.pyplot as plt
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import GWR
g = gpd.read_file("dataset/G_utm.shp")
cent = g.geometry.centroid
coords = np.c_[cent.x, cent.y]
y = g.PctBach.values.reshape(-1, 1)
Xr = g[["PctFB", "PctBlack", "PctRural"]].values
Xs = (Xr - Xr.mean(0)) / Xr.std(0)
sel = Sel_BW(coords, y, Xs, fixed=True, kernel="gaussian")
best_bw = sel.search()
res = GWR(coords, y, Xs, bw=best_bw, fixed=True, kernel="gaussian").fit()
cands = np.linspace(0.5 * best_bw, 2.5 * best_bw, 5)
fig, ax = plt.subplots(2, 3, figsize=(16, 10))
for a, bw in zip(ax.flat, list(cands) + [best_bw]):
    r = GWR(coords, y, Xs, bw=bw, fixed=True, kernel="gaussian").fit()
    g.assign(beta_rural=r.params[:, 3]).plot("beta_rural", ax=a, cmap="RdBu_r", legend=True)
    a.set_title(f"bw = {bw/1000:.0f} km"); a.set_axis_off()
plt.savefig("pred_results/GA_GWR.png")
