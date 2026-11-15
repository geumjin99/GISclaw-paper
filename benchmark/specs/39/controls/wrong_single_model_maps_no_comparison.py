"""Wrong: the Sel_BW optimum (117 NN) is fitted correctly, but the figure shows only that one model's
three slope maps; no second model or bandwidth is plotted, so their differences are not shown.
"""
import geopandas as gpd, numpy as np, matplotlib.pyplot as plt
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import GWR
ga = gpd.read_file("dataset/G_utm.shp")
y = ga[["PctBach"]].to_numpy(float)
X = ga[["PctFB", "PctBlack", "PctRural"]].to_numpy(float)
uv = np.column_stack([ga.X, ga.Y])
bw = float(Sel_BW(uv, y, X).search(criterion="AICc"))
res = GWR(uv, y, X, bw=bw).fit()
for j, n in enumerate(["Intercept", "PctFB", "PctBlack", "PctRural"]):
    ga[f"b_{n}"] = res.params[:, j]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, n in zip(axes, ["PctFB", "PctBlack", "PctRural"]):
    ga.plot(f"b_{n}", ax=ax, cmap="RdBu_r", legend=True, edgecolor="grey", linewidth=0.2)
    ax.set_title(f"local {n} coefficient (optimal bw={bw:.0f} NN)"); ax.set_axis_off()
fig.suptitle(f"GWR of PctBach, optimal bandwidth {bw:.0f} NN (AICc {res.aicc:.1f})")
fig.savefig("pred_results/GA_GWR.png")
