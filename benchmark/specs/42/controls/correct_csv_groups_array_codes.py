"""Correct: district medians from the CSV's neighbourhood_group_cleansed, gaps filled from Queen
neighbours; the LISA codes are held only in a numpy array (+1 hot, -1 cold, 0 other).
"""
import numpy as np, pandas as pd, geopandas as gpd
from libpysal.weights import Queen
from esda.moran import Moran_Local
import matplotlib.pyplot as plt

def lisa_codes():
    nb = gpd.read_file("dataset/berlin-neighbourhoods.geojson")
    df = pd.read_csv("dataset/berlin-listings.csv", usecols=["price", "neighbourhood_group_cleansed"])
    grp_med = df.groupby("neighbourhood_group_cleansed").price.median()
    vals = nb.neighbourhood_group.map(grp_med).to_numpy(float)
    W = Queen.from_dataframe(nb, use_index=False)
    for i in np.flatnonzero(np.isnan(vals)):
        vals[i] = np.nanmean(vals[W.neighbors[i]])
    vals[np.isnan(vals)] = np.nanmean(grp_med.values)  # the two group-less polygons only touch each other
    W.transform = "R"
    ml = Moran_Local(vals, W, permutations=4999, seed=2024)
    codes = np.zeros(len(nb), int)
    codes[(ml.p_sim < 0.05) & (ml.q == 1)] = 1
    codes[(ml.p_sim < 0.05) & (ml.q == 3)] = -1
    colors = np.array(["#d7301f" if c == 1 else "#2c7fb8" if c == -1 else "#dddddd" for c in codes])
    ax = nb.plot(color=colors, edgecolor="white", figsize=(9, 9))
    ax.set_title("LISA hot (red) and cold (blue) spots"); ax.axis("off")
    plt.savefig("pred_results/moran_local.png")

lisa_codes()
