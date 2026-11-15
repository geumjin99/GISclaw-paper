"""Correct: everything inside main(); the result is written only to a shapefile (no figure)."""
import numpy as np, pandas as pd, geopandas as gpd
from sklearn.cluster import KMeans
from scipy.stats import zscore
RISKS = {"transmission": ["PopDensity", "SpInterIdx"], "susceptibility": ["PopDensity", "Seniors60t", "spending_c", "keyfacts_p"],
         "scarcity": ["Seniors60t", "SUM_Beds"], "exposure": ["RelativeCa"]}
def simrank(df, tgt, cols):
    X = df[cols].astype(float); mu, sd = X.mean(), X.std(ddof=0)
    d = (((X - mu) / sd - (tgt[cols].astype(float).iloc[0] - mu) / sd) ** 2).sum(axis=1)
    return d.rank(method="first").astype(int)
def main():
    hkg = gpd.read_file("dataset/hkg_constituency.shp"); tgt = gpd.read_file("dataset/target_risk.shp")
    for k, cols in RISKS.items():
        hkg[k + "_r"] = 432 - simrank(hkg, tgt, cols)
    f = [k + "_r" for k in RISKS]
    Z = pd.DataFrame(zscore(hkg[f].astype(float)))
    seeds = np.flatnonzero(hkg.SEEDS == 1)
    hkg["profile"] = KMeans(len(seeds), init=Z.values[seeds], n_init=1).fit_predict(Z) + 1
    hkg.to_file("pred_results/covid_risk_profiles.shp")
main()
