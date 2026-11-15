"""Correct: Sel_BW optimum; the figure is a bar chart of AICc by bandwidth (no maps)."""
import geopandas as gpd, numpy as np, matplotlib.pyplot as plt
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import GWR
ga = gpd.read_file("dataset/G_utm.shp")
y = ga[["PctBach"]].to_numpy(float)
X = ga[["PctFB", "PctBlack", "PctRural"]].to_numpy(float)
uv = np.column_stack([ga.X, ga.Y])
opt = float(Sel_BW(uv, y, X).search(criterion="AICc"))
res = GWR(uv, y, X, bw=opt).fit()
for j, n in enumerate(["Intercept", "PctFB", "PctBlack", "PctRural"]):
    ga[f"b_{n}"] = res.params[:, j]
bws = sorted({50, 75, 100, int(round(opt)), 125, 150})
aiccs = [GWR(uv, y, X, bw=b).fit().aicc for b in bws]
fig, ax = plt.subplots(figsize=(9, 6))
ax.bar(bws, aiccs, width=8, color=["tab:red" if b == round(opt) else "tab:grey" for b in bws])
ax.set_ylim(min(aiccs) - 5, max(aiccs) + 5)
ax.set_xlabel("adaptive bandwidth (NN)"); ax.set_ylabel("AICc"); ax.set_title(f"GWR AICc by bandwidth (optimum {opt:.0f})")
fig.savefig("pred_results/GA_GWR.png")
