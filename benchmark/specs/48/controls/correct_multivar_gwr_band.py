# Correct: Gi* with a fixed distance band (libpysal DistanceBand, min threshold); GWR with two covariates
# (grade + household loan ratio), adaptive bisquare 22 NN; results only in arrays inside a function.
import numpy as np, geopandas as gpd, libpysal
from esda.getisord import G_Local
from mgwr.gwr import GWR
import matplotlib.pyplot as plt

def run():
    t = gpd.read_file("dataset/loan_data.shp")
    t = t.loc[t["AcceptedLo"] >= 30].copy()
    c = np.column_stack([t.centroid.x, t.centroid.y])
    thr = libpysal.weights.min_threshold_distance(c)
    wb = libpysal.weights.DistanceBand(c, threshold=thr, binary=True, silence_warnings=True)
    gi = G_Local(t["AveInteres"].to_numpy(), wb, star=True, permutations=0)
    zs = gi.Zs
    X = t[["AveLoanGra", "pHHLoanRat"]].to_numpy(float)
    X = (X - X.mean(0)) / X.std(0)
    fit = GWR(c, t[["AveInteres"]].to_numpy(float), X, bw=22, kernel="bisquare", fixed=False).fit()
    b_grade = fit.params[:, 1]
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(c[:, 0], c[:, 1], c=zs, cmap="RdBu_r"); ax[1].scatter(c[:, 0], c[:, 1], c=b_grade)
    t.to_file("pred_results/selected_tracts.gpkg", driver="GPKG")
    fig.savefig("pred_results/hotspots_gwr.png")

run()
