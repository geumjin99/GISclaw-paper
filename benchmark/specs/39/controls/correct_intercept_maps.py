"""Correct: Sel_BW optimum; the figure compares bandwidths through the local intercept surface only
(the slopes are kept in the table, not drawn).
"""
import geopandas as gpd, numpy as np, matplotlib.pyplot as plt
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import GWR
ga = gpd.read_file("dataset/G_utm.shp")
y = ga[["PctBach"]].to_numpy(float)
X = ga[["PctFB", "PctBlack", "PctRural"]].to_numpy(float)
uv = np.column_stack([ga.X, ga.Y])
opt = float(Sel_BW(uv, y, X).search(criterion="AICc"))
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, b in zip(axes, [50, opt, 150]):
    r = GWR(uv, y, X, bw=b).fit()
    for j, n in enumerate(["Intercept", "PctFB", "PctBlack", "PctRural"]):
        ga[f"bw{b:.0f}_{n}"] = r.params[:, j]
    ga.plot(f"bw{b:.0f}_Intercept", ax=ax, cmap="viridis", legend=True); ax.set_title(f"local intercept, bw={b:.0f}")
fig.savefig("pred_results/GA_GWR.png")
