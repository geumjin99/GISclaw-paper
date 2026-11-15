# Wrong: the hot spots are computed on all 878 tracts and the selection is applied afterwards;
# the instruction asks for hot spots on the selected tracts.
import os, numpy as np, geopandas as gpd, libpysal
from esda.getisord import G_Local
from mgwr.gwr import GWR
import matplotlib.pyplot as plt
os.makedirs("pred_results", exist_ok=True)
allt = gpd.read_file("dataset/loan_data.shp")
# hot spots computed on ALL tracts (incl. those with < 30 loans), then the selection is applied
wk = libpysal.weights.KNN.from_dataframe(allt, k=8)
allt["GiZ"] = G_Local(allt.AveInteres.values, wk, star=True, permutations=0).Zs
sel = allt[allt.AcceptedLo >= 30].reset_index(drop=True)
xy = np.c_[sel.centroid.x, sel.centroid.y]
res = GWR(xy, sel[["AveInteres"]].values, sel[["AveLoanGra"]].values, bw=22, fixed=False, kernel="bisquare").fit()
sel["coef_grade"] = res.params[:, 1]; sel["localR2"] = res.localR2.ravel()
sel.to_file("pred_results/loan_gwr.shp")
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
sel.plot("GiZ", cmap="RdBu_r", ax=ax[0], legend=True); sel.plot("coef_grade", ax=ax[1], legend=True)
plt.savefig("pred_results/loan_hotspots_gwr.png")
