"""Correct: coarse hand grid search 50..130 step 40 on AICc (no Sel_BW) -> 130 NN, about 1 AICc unit
above the Sel_BW optimum (117) on a flat AICc curve.
"""
import geopandas as gpd, numpy as np, matplotlib.pyplot as plt
from mgwr.gwr import GWR
ga = gpd.read_file("dataset/G_utm.shp")
y = ga[["PctBach"]].to_numpy(float)
X = ga[["PctFB", "PctBlack", "PctRural"]].to_numpy(float)
uv = np.column_stack([ga.X, ga.Y])
aicc = {}
for bw in range(50, 131, 40):
    aicc[bw] = GWR(uv, y, X, bw=bw).fit().aicc
best_bw = min(aicc, key=aicc.get)
res = GWR(uv, y, X, bw=best_bw).fit()
for j, n in enumerate(["Intercept", "PctFB", "PctBlack", "PctRural"]):
    ga[f"b_{n}"] = res.params[:, j]
print("best bandwidth", best_bw)
# maps only (no AICc plot): the bandwidth evidence must come from the search + best_bw;
# the figure compares the PctRural surface of the three grid bandwidths
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, b in zip(axes, sorted(aicc)):
    ga.assign(b_rural=GWR(uv, y, X, bw=b).fit().params[:, 3]).plot("b_rural", ax=ax, cmap="RdBu_r", legend=True)
    ax.set_title(f"PctRural coefficient, bw={b}" + (" (best)" if b == best_bw else ""))
fig.savefig("pred_results/GA_GWR.png")
