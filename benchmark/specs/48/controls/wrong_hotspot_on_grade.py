# Wrong: Gi* is computed on the loan grade (AveLoanGra) instead of the interest rate (AveInteres).
import os, numpy as np, geopandas as gpd, libpysal
from esda.getisord import G_Local
from mgwr.gwr import GWR
import matplotlib.pyplot as plt
os.makedirs("pred_results", exist_ok=True)
allt = gpd.read_file("dataset/loan_data.shp")
sel = allt[allt.AcceptedLo >= 30].reset_index(drop=True)
wk = libpysal.weights.KNN.from_dataframe(sel, k=8)
sel["GiZ"] = G_Local(sel.AveLoanGra.values, wk, star=True, permutations=0).Zs   # wrong field
xy = np.c_[sel.centroid.x, sel.centroid.y]
res = GWR(xy, sel[["AveInteres"]].values, sel[["AveLoanGra"]].values, bw=22, fixed=False, kernel="bisquare").fit()
sel["coef_grade"] = res.params[:, 1]; sel["localR2"] = res.localR2.ravel()
sel.to_file("pred_results/loan_gwr.shp")
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
sel.plot("GiZ", cmap="RdBu_r", ax=ax[0], legend=True); sel.plot("coef_grade", ax=ax[1], legend=True)
plt.savefig("pred_results/loan_hotspots_gwr.png")
