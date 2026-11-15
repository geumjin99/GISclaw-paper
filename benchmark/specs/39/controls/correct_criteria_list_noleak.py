"""Correct: compares the optimal bandwidths chosen by four criteria; the bandwidths live only in a
list / dict of results (no loop variable is left over).
"""
import geopandas as gpd, numpy as np, matplotlib.pyplot as plt
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import GWR

def run():
    ga = gpd.read_file("dataset/G_utm.shp")
    y = ga[["PctBach"]].to_numpy(float)
    X = ga[["PctFB", "PctBlack", "PctRural"]].to_numpy(float)
    uv = np.column_stack([ga.X, ga.Y])
    crits = ["AICc", "AIC", "BIC", "CV"]
    bws = [float(Sel_BW(uv, y, X).search(criterion=c)) for c in crits]
    fits = [GWR(uv, y, X, bw=bw_).fit() for bw_ in bws]
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    list(map(lambda t: t[0].scatter(uv[:, 0], uv[:, 1], c=t[1].params[:, 3], cmap="coolwarm", s=40), zip(axes, fits)))
    list(map(lambda t: t[0].set_title(f"{t[1]} optimum: {t[2]:.0f} NN"), zip(axes, crits, bws)))
    fig.savefig("pred_results/GA_GWR.png")

run()
