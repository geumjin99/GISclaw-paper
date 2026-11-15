"""Reference solution for task 25: COVID-19 risk profiles of Hong Kong constituencies.

Open-source equivalent of ArcGIS Similarity Search (ATTRIBUTE_VALUES, MOST_SIMILAR,
Number_Of_Results=0) + Add Join / Calculate Field (432 - SIMRANK) + Multivariate Clustering
(K-means, user-defined seed locations from field SEEDS)."""
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RISKS = {
    "trans_risk": ["PopDensity", "SpInterIdx"],                               # transmission
    "susc_risk": ["PopDensity", "Seniors60t", "spending_c", "keyfacts_p"],    # susceptibility
    "insr_risk": ["Seniors60t", "SUM_Beds"],                                   # insufficient resources
    "expo_risk": ["RelativeCa"],                                               # exposure
}


def similarity_search(candidates, target, fields):
    """ArcGIS Similarity Search on attribute values: z-standardise every attribute (mean/std
    of the candidates), sum of squared differences to the target, SIMRANK 1 = most similar."""
    X = candidates[fields].astype(float).to_numpy()
    T = target[fields].astype(float).to_numpy()[0]
    mu, sd = X.mean(axis=0), X.std(axis=0)
    ssd = (((X - mu) / sd - (T - mu) / sd) ** 2).sum(axis=1)
    res = pd.DataFrame({"ID": candidates["ID"].to_numpy(), "SIMINDEX": ssd})
    res["SIMRANK"] = res["SIMINDEX"].rank(method="first").astype(int)
    return res


def main():
    hkg = gpd.read_file("dataset/hkg_constituency.shp")
    target = gpd.read_file("dataset/target_risk.shp")
    n = len(hkg)                                    # 431 -> 432 - SIMRANK gives 431 ... 1

    for field, attrs in RISKS.items():
        sim = similarity_search(hkg, target, attrs)
        joined = hkg.merge(sim[["ID", "SIMRANK"]], on="ID", how="left")      # Add Join on ID
        hkg[field] = (n + 1 - joined["SIMRANK"]).to_numpy()                 # 432 - SIMRANK
        # (Remove Join: the SIMRANK column is not kept)

    # Multivariate Clustering: K-means on standardised analysis fields; initial centres are the
    # features flagged in SEEDS (3 seeds -> 3 clusters)
    fields = list(RISKS)
    F = hkg[fields].astype(float).to_numpy()
    Z = (F - F.mean(axis=0)) / F.std(axis=0)
    seeds = np.flatnonzero(hkg["SEEDS"].to_numpy() == 1)
    km = KMeans(n_clusters=len(seeds), init=Z[seeds], n_init=1, random_state=0).fit(Z)
    hkg["CLUSTER_ID"] = km.labels_ + 1
    profile = hkg.groupby("CLUSTER_ID")[fields].mean().round(1)
    profile["n"] = hkg.groupby("CLUSTER_ID").size()
    print(profile)

    hkg.to_file("pred_results/risk_profiles.shp")
    risk_profiles = hkg

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    titles = {"trans_risk": "Transmission risk", "susc_risk": "Susceptibility risk",
              "insr_risk": "Healthcare-resource scarcity risk", "expo_risk": "Exposure risk"}
    for ax, f in zip(axes.flat[:4], fields):
        hkg.plot(column=f, cmap="OrRd", legend=True, ax=ax)
        ax.set_title(titles[f] + " (432 - SIMRANK)")
        ax.set_axis_off()
    hkg.plot(column="CLUSTER_ID", categorical=True, cmap="Set1", legend=True, ax=axes.flat[4])
    axes.flat[4].set_title("COVID-19 risk profiles (K-means, SEEDS)")
    axes.flat[4].set_axis_off()
    axes.flat[5].axis("off")
    axes.flat[5].table(cellText=profile.values, colLabels=list(profile.columns),
                       rowLabels=[f"cluster {i}" for i in profile.index], loc="center")
    fig.tight_layout()
    fig.savefig("pred_results/covid_risk.png", dpi=110)


main()
