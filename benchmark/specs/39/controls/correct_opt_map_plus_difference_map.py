"""Correct: the optimal model's PctRural map next to a difference map (PctRural coefficient at 50 NN
minus the optimum); the comparison is drawn as a difference.
"""
import geopandas as gpd, numpy as np, matplotlib.pyplot as plt
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import GWR
ga = gpd.read_file("dataset/G_utm.shp")
y = ga[["PctBach"]].to_numpy(float)
X = ga[["PctFB", "PctBlack", "PctRural"]].to_numpy(float)
uv = np.column_stack([ga.X, ga.Y])
bw = Sel_BW(uv, y, X).search()
b_opt = GWR(uv, y, X, bw=bw).fit().params[:, 3]
b_50 = GWR(uv, y, X, bw=50).fit().params[:, 3]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ga.assign(v=b_opt).plot("v", ax=axes[0], cmap="viridis", legend=True); axes[0].set_title(f"PctRural coefficient, optimal bw={bw:.0f}")
ga.assign(v=b_50 - b_opt).plot("v", ax=axes[1], cmap="RdBu_r", legend=True); axes[1].set_title(f"difference: bw=50 minus bw={bw:.0f}")
fig.savefig("pred_results/GA_GWR.png")
