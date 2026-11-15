"""Correct: maps of the optimal GWR's three slopes plus a bar chart comparing the AICc and R2 of GWR
(Sel_BW optimum) and MGWR; the models are compared through their diagnostics.
"""
import geopandas as gpd, numpy as np, matplotlib.pyplot as plt
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import GWR, MGWR
ga = gpd.read_file("dataset/G_utm.shp")
y = ga[["PctBach"]].to_numpy(float)
X = ga[["PctFB", "PctBlack", "PctRural"]].to_numpy(float)
uv = np.column_stack([ga.X, ga.Y])
bw = Sel_BW(uv, y, X).search()
gwr = GWR(uv, y, X, bw=bw).fit()
msel = Sel_BW(uv, y, X, multi=True); msel.search()
mgwr = MGWR(uv, y, X, msel).fit()
fig, axes = plt.subplots(1, 5, figsize=(26, 6))
for ax, j, n in zip(axes[:3], [1, 2, 3], ["PctFB", "PctBlack", "PctRural"]):
    ga.assign(b=gwr.params[:, j]).plot("b", ax=ax, cmap="RdBu_r", legend=True); ax.set_title(f"GWR {n} (bw={bw:.0f})"); ax.set_axis_off()
axes[3].bar(["GWR", "MGWR"], [gwr.aicc, mgwr.aicc]); axes[3].set_ylim(min(gwr.aicc, mgwr.aicc) - 10, max(gwr.aicc, mgwr.aicc) + 5); axes[3].set_title("AICc")
axes[4].bar(["GWR", "MGWR"], [gwr.R2, mgwr.R2]); axes[4].set_title("R2")
fig.savefig("pred_results/GA_GWR.png")
