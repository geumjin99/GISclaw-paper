"""Wrong: the analysis is the reference solution, but every geographic deliverable file is rewritten
without a CRS; geographic deliverables must carry a CRS.
"""
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


# ---------------------------------------------------------------------------------------------
# NEGATIVE CONTROL: the analysis above is the reference solution, but every geographic
# deliverable file is then written without a CRS. Expected: wrong.
def _write_without_crs(path):
    """Rewrite one file in place so that it carries no CRS. Returns a short description.
    .shp: drop the .prj sidecar. .tif/.tiff: rewrite without CRS (and drop .aux.xml, which can
    hold one). .gpkg: rewrite every spatial layer with crs=None. .geojson/.json: write projected
    coordinates (UTM if the data are lon/lat) without a 'crs' member - a lon/lat GeoJSON
    without 'crs' is WGS 84 by definition (RFC 7946), so dropping the member alone would not
    remove the CRS."""
    import os
    low = path.lower()
    stem, _ = os.path.splitext(path)
    if low.endswith(".shp"):
        d, b = os.path.split(stem)
        gone = []
        for n in os.listdir(d or "."):
            if n.lower() == (b + ".prj").lower():
                os.remove(os.path.join(d, n))
                gone.append(n)
        return f"removed {gone}"
    if low.endswith((".tif", ".tiff")):
        import rasterio
        with rasterio.open(path) as r:
            prof, data = r.profile.copy(), r.read()
            tags = r.tags()
        prof.pop("crs", None)
        tmp = stem + ".__nocrs__" + os.path.splitext(path)[1]
        with rasterio.open(tmp, "w", **prof) as w:
            w.write(data)
            if tags:
                w.update_tags(**tags)
        os.replace(tmp, path)
        if os.path.exists(path + ".aux.xml"):
            os.remove(path + ".aux.xml")
        return "raster rewritten without CRS"
    if low.endswith(".gpkg"):
        import geopandas as gpd
        import pyogrio
        tmp = stem + ".__nocrs__.gpkg"
        for lyr, gt in pyogrio.list_layers(path):
            g = gpd.read_file(path, layer=lyr)
            if gt is not None:
                g = g.set_crs(None, allow_override=True)
            g.to_file(tmp, layer=lyr, driver="GPKG")
        os.replace(tmp, path)
        return "gpkg rewritten without CRS"
    if low.endswith((".geojson", ".json")):
        import geopandas as gpd
        g = gpd.read_file(path)
        if g.crs is None or g.crs.is_geographic:
            if g.crs is None:
                g = g.set_crs(4326)
            g = g.to_crs(g.estimate_utm_crs())
        g = g.set_crs(None, allow_override=True)
        os.remove(path)
        g.to_file(path, driver="GeoJSON")
        return "GeoJSON rewritten in projected coordinates without 'crs'"
    return "not a geographic file"


def _strip_deliverable_crs(patterns):
    import fnmatch
    import os
    for n in sorted(os.listdir("pred_results")):
        low = n.lower()
        if low.endswith(('.shp', '.geojson', '.gpkg', '.json', '.tif', '.tiff')) and any(fnmatch.fnmatchcase(low, p.lower()) for p in patterns):
            if low.endswith(".json"):
                try:
                    import json
                    if json.load(open(os.path.join("pred_results", n))).get("type") is None:
                        continue
                except Exception:
                    continue
            print("no-CRS control:", n, _write_without_crs(os.path.join("pred_results", n)))


_strip_deliverable_crs(['*.geojson', '*.gpkg', '*.shp'])
