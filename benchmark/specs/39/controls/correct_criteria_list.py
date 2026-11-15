"""Correct: compares the optimal bandwidths chosen by four criteria; the bandwidths are kept only in a list."""
import geopandas as gpd, numpy as np, matplotlib.pyplot as plt
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import GWR
ga = gpd.read_file("dataset/G_utm.shp")
y = ga[["PctBach"]].to_numpy(float)
X = ga[["PctFB", "PctBlack", "PctRural"]].to_numpy(float)
uv = np.column_stack([ga.X, ga.Y])
crits = ["AICc", "AIC", "BIC", "CV"]
bws = [Sel_BW(uv, y, X).search(criterion=c) for c in crits]
fits = [GWR(uv, y, X, bw=b).fit() for b in bws]
fig, axes = plt.subplots(1, 4, figsize=(20, 6))
for ax, f, c, b in zip(axes, fits, crits, bws):
    sc = ax.scatter(uv[:, 0], uv[:, 1], c=f.params[:, 3], cmap="coolwarm", s=40)
    ax.set_title(f"{c}: bw={b:.0f}  R2={f.R2:.2f}"); ax.set_aspect("equal"); fig.colorbar(sc, ax=ax, shrink=0.6)
fig.suptitle("Local PctRural coefficient at criterion-optimal bandwidths")
fig.savefig("pred_results/GA_GWR.png")
