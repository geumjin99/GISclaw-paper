"""Reference solution for task 39: Geographically Weighted Regression of PctBach in Georgia
counties (mgwr).

y = PctBach; X = PctFB, PctBlack, PctRural (+ intercept); coordinates = county centroids
(X, Y columns, UTM metres). Optimal bandwidth: golden-section search, adaptive bisquare
kernel, AICc (mgwr defaults) -> 117 nearest neighbours. The effect of the bandwidth is
shown by mapping the local PctRural coefficient for a range of bandwidths (25..150, as in
the benchmark's figure) next to the optimal-bandwidth model, plus the AICc profile.

Adapted from the reference script of GeoAnalystBench (Zhang et al. 2025, Apache-2.0).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

os.makedirs("pred_results", exist_ok=True)

georgia = gpd.read_file("dataset/G_utm.shp")
y = georgia["PctBach"].values.reshape((-1, 1))
X = georgia[["PctFB", "PctBlack", "PctRural"]].values
coords = list(zip(georgia["X"], georgia["Y"]))

selector = Sel_BW(coords, y, X)               # adaptive bisquare, AICc
gwr_bw = float(selector.search(criterion="AICc"))
gwr_results = GWR(coords, y, X, bw=gwr_bw).fit()
print("optimal bandwidth:", gwr_bw, "AICc", gwr_results.aicc, "R2", gwr_results.R2)

names = ["Intercept", "PctFB", "PctBlack", "PctRural"]
coef_table = pd.DataFrame({"AreaKey": georgia["AreaKey"].values})
for j, n in enumerate(names):
    coef_table[f"opt_{n}"] = gwr_results.params[:, j]

bandwidths = [25, 50, 75, 100, 125, 150]
aicc = {}
for bw in bandwidths:
    res = GWR(coords, y, X, bw=bw).fit()
    aicc[bw] = res.aicc
    for j, n in enumerate(names):
        coef_table[f"bw{bw}_{n}"] = res.params[:, j]
# optima of the other common kernel choices (legitimate alternatives, kept for the checks)
for lab, kw in (("fixedbisq", dict(fixed=True, kernel="bisquare")), ("adaptgauss", dict(kernel="gaussian"))):
    b_alt = float(Sel_BW(coords, y, X, **kw).search())
    res = GWR(coords, y, X, bw=b_alt, **kw).fit()
    for j, n in enumerate(names):
        coef_table[f"{lab}_{n}"] = res.params[:, j]
bw_profile = pd.DataFrame({"bandwidth": list(range(30, 160, 5))})
bw_profile["AICc"] = [GWR(coords, y, X, bw=b).fit().aicc for b in bw_profile["bandwidth"]]

for c in coef_table.columns:
    if c != "AreaKey":
        georgia[c] = coef_table[c].values
georgia.to_file("pred_results/GA_GWR_coefficients.geojson", driver="GeoJSON")

panels = [f"bw{b}_PctRural" for b in bandwidths] + ["opt_PctRural"]
vmin = min(georgia[p].min() for p in panels)
vmax = max(georgia[p].max() for p in panels)
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
for ax, p in zip(axes.flat, panels):
    georgia.plot(p, ax=ax, cmap="RdBu_r", vmin=vmin, vmax=vmax, edgecolor="grey", linewidth=0.2)
    bwlab = p.split("_")[0].replace("bw", "")
    ax.set_title(f"Bandwidth {bwlab} NN" if p != "opt_PctRural" else f"Optimal bandwidth {gwr_bw:.0f} NN (AICc)")
    ax.set_axis_off()
ax = axes.flat[-1]
ax.plot(bw_profile["bandwidth"], bw_profile["AICc"], "-k")
ax.axvline(gwr_bw, color="r", ls="--", label=f"optimal = {gwr_bw:.0f}")
ax.set_xlabel("adaptive bandwidth (nearest neighbours)"); ax.set_ylabel("AICc"); ax.legend()
ax.set_title("AICc vs bandwidth")
sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin, vmax))
fig.colorbar(sm, ax=axes[:, :3].ravel().tolist(), shrink=0.6, label="local coefficient of PctRural")
fig.suptitle("GWR of PctBach on PctFB, PctBlack, PctRural - effect of bandwidth")
plt.savefig("pred_results/GA_GWR.png", dpi=150)
