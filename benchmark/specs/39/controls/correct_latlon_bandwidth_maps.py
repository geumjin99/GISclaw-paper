"""Correct: lon/lat coordinates (mgwr computes great-circle distances), Sel_BW optimum; the PctRural
coefficient is mapped at bw/2, the optimum and 1.3*bw.
"""
import geopandas as gpd, numpy as np, pandas as pd, matplotlib.pyplot as plt
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import GWR
shp = gpd.read_file("dataset/G_utm.shp")
data = pd.read_csv("dataset/GData_utm.csv")
coords = list(zip(data.Longitud, data.Latitude))
cols = ["PctFB", "PctBlack", "PctRural"]
y = data[["PctBach"]].values; X = data[cols].values
bw = Sel_BW(coords, y, X).search()
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, b in zip(axes, [round(bw / 2), bw, min(round(bw * 1.3), 158)]):
    res = GWR(coords, y, X, bw=b).fit()
    out = shp[["AreaKey", "geometry"]].merge(pd.DataFrame(res.params, columns=["b0"] + [f"b_{c}" for c in cols]).assign(AreaKey=data.AreaKey.values), on="AreaKey")
    out.plot("b_PctRural", ax=ax, cmap="viridis", legend=True); ax.set_title(f"PctRural, bw={b:.0f} (AICc {res.aicc:.1f})")
plt.savefig("pred_results/GA_GWR.png")
