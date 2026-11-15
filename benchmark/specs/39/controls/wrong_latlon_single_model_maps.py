"""Wrong: lon/lat coordinates are acceptable, but the figure shows only the optimal model's three
slope maps; no second model or bandwidth is drawn, so no comparison is plotted.
"""
import geopandas as gpd, numpy as np, pandas as pd, matplotlib.pyplot as plt
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import GWR
shp = gpd.read_file("dataset/G_utm.shp")
tab = pd.read_csv("dataset/GData_utm.csv")
data = tab
coords = list(zip(data.Longitud, data.Latitude))
cols = ["PctFB", "PctBlack", "PctRural"]
y = data[["PctBach"]].values; X = data[cols].values
sel = Sel_BW(coords, y, X)
bw = sel.search()
res = GWR(coords, y, X, bw=bw).fit()
out = shp[["AreaKey", "geometry"]].merge(pd.DataFrame(res.params, columns=["b0"] + [f"b_{c}" for c in cols]).assign(AreaKey=data.AreaKey.values), on="AreaKey")
fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 6))
for ax, c in zip(axes, cols):
    out.plot(f"b_{c}", ax=ax, cmap="viridis", legend=True); ax.set_title(f"{c} (bw={bw:.0f})")
plt.savefig("pred_results/GA_GWR.png")
