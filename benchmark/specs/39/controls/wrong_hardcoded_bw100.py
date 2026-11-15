"""Wrong: the bandwidth is hard-coded to 100 nearest neighbours without a search. Although 100 NN is
within 1 AICc unit of the optimum, the task asks to search for the optimal bandwidth.
"""
import geopandas as gpd, numpy as np, matplotlib.pyplot as plt
from mgwr.gwr import GWR
ga = gpd.read_file("dataset/G_utm.shp")
y = ga[["PctBach"]].to_numpy(float)
X = ga[["PctFB", "PctBlack", "PctRural"]].to_numpy(float)
uv = np.column_stack([ga.X, ga.Y])
res = GWR(uv, y, X, bw=100).fit()
for j, n in enumerate(["Intercept", "PctFB", "PctBlack", "PctRural"]):
    ga[f"b_{n}"] = res.params[:, j]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, n in zip(axes, ["PctFB", "PctBlack", "PctRural"]):
    ga.plot(f"b_{n}", ax=ax, cmap="RdBu_r", legend=True); ax.set_title(f"{n} coefficient (bw=100)")
fig.savefig("pred_results/GA_GWR.png")
