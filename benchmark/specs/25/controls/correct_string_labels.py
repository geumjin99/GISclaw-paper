"""Correct: risk scores and seeded K-means; the cluster is stored as descriptive text labels."""
import numpy as np, pandas as pd, geopandas as gpd
from sklearn.cluster import KMeans
from scipy.stats import zscore
RISKS = {"transmission": ["PopDensity", "SpInterIdx"], "susceptibility": ["PopDensity", "Seniors60t", "spending_c", "keyfacts_p"],
         "scarcity": ["Seniors60t", "SUM_Beds"], "exposure": ["RelativeCa"]}
def simrank(df, tgt, cols):
    X = df[cols].astype(float); mu, sd = X.mean(), X.std(ddof=0)
    d = (((X - mu) / sd - (tgt[cols].astype(float).iloc[0] - mu) / sd) ** 2).sum(axis=1)
    return d.rank(method="first").astype(int)
hkg = gpd.read_file("dataset/hkg_constituency.shp"); tgt = gpd.read_file("dataset/target_risk.shp")
for k, cols in RISKS.items():
    hkg[k] = 432 - simrank(hkg, tgt, cols)
Z = pd.DataFrame(zscore(hkg[list(RISKS)].astype(float)))
seeds = np.flatnonzero(hkg.SEEDS == 1)
lab = KMeans(len(seeds), init=Z.values[seeds], n_init=1).fit_predict(Z)
hkg["profile"] = ["Profile " + "ABC"[i] for i in lab]
hkg.to_file("pred_results/hkg_risk.gpkg", driver="GPKG")
ax = hkg.plot(column="profile", categorical=True, legend=True); ax.figure.savefig("pred_results/risk_clusters.png")
