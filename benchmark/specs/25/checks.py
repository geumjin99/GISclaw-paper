"""T25 - COVID-19 risk profiles, Hong Kong (431 constituencies).

Task core: four Similarity Searches against the worst-case target (attribute sets given in
the task), each converted to a risk score 432 - SIMRANK and joined on ID, then K-means
multivariate clustering of the four scores (initialised from SEEDS: 3 seeds -> 3 clusters).

Correctness (required):
  1. an output exists (a map PNG that is not blank, or a vector/table file with the result);
  2-5. each of the four risk scores exists as a per-constituency column whose ranks agree with
     the gold score (Spearman >= 0.95, aligned on ID). Raw input attributes are excluded from
     the search. Unstandardised distances (susceptibility rho 0.52) or a reversed score
     (rank instead of 432 - rank) fail;
  6. a cluster-label column (2-10 clusters; numeric or text/categorical labels) whose partition is a near-optimal K-means
     partition of the four gold risk scores: its between-cluster share of variance (R2, on
     standardised gold scores) >= 0.90 x the best K-means R2 for the same k. Clustering on raw
     attributes or on a subset of the risks gives <= 0.80.
  7. (required: the domain knowledge prescribes "K means as clustering
     method, SEEDS as initialization field") method fidelity: some 3-class cluster column
     agrees with the SEEDS-initialised gold (ARI >= 0.8). The 3 seeds fix k = 3 and, on the
     correct scores, the seeded partition (z-scores, ddof 0/1, min-max or raw scores all give
     ARI 1.0); k-means++ partitions of equal explained variance reach only ARI ~0.33.
"""
import glob
import os

import numpy as np
import pandas as pd

from checklib import result, png_not_blank, best_matching_column, key_values

N = 431
RISKS = ["trans_risk", "susc_risk", "insr_risk", "expo_risk"]
LABELS = {"trans_risk": "transmission", "susc_risk": "susceptibility", "insr_risk": "resource scarcity",
          "expo_risk": "exposure"}
INPUT_COLS = {"TotalPop", "PopDensity", "Seniors60t", "spending_c", "keyfacts_p", "SUM_Beds", "SpInterIdx",
              "RelativeCa", "SEEDS", "Shape_Leng", "Shape_Area", "index", "level_0", "index_right", "OBJECTID", "FID"}


def _gold(ctx):
    import geopandas as gpd
    g = gpd.read_file(ctx.gold("risk_profiles.shp"))
    return g.set_index("ID")


def _r2(labels, Z):
    tot = ((Z - Z.mean(0)) ** 2).sum()
    w = sum(((Z[labels == c] - Z[labels == c].mean(0)) ** 2).sum() for c in np.unique(labels))
    return 1 - w / tot


def checks(ctx):
    import geopandas as gpd
    out = []
    pngs = [p for p in ctx.pred_files("*.png")]
    nb = any(png_not_blank(ctx.pred(p))[0] for p in pngs)
    vecs = [p for p in ctx.pred_files("*") if p.lower().endswith((".shp", ".gpkg", ".geojson", ".csv"))]
    out.append(result("output written (non-blank map or result file)", nb or bool(vecs), value=pngs + vecs))

    try:
        gold = _gold(ctx)
        inputs = gpd.read_file(os.path.join(ctx.inputs_dir, "hkg_constituency.shp"))
    except Exception as e:
        for f in RISKS:
            out.append(result(f"{LABELS[f]} risk score matches gold (Spearman)", False, note="gold missing"))
        out.append(result("clustering of the four risk scores", False))
        return out

    for f in RISKS:
        excl = INPUT_COLS | {"CLUSTER_ID", "cluster", "Cluster", "SIMRANK"}
        col, rho, where, s = best_matching_column(ctx, gold[f].astype(float), inputs, "ID", N, exclude=excl)
        out.append(result(f"{LABELS[f]} risk score matches gold (Spearman)", rho >= 0.95,
                          value=round(rho, 4) if col else None, threshold=0.95,
                          note=f"{col} in {where}" if col else "no per-constituency score found"))

    # clustering
    F = gold[RISKS].astype(float).to_numpy()
    Z = (F - F.mean(0)) / F.std(0)
    gold_lab = gold["CLUSTER_ID"].to_numpy()
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    best = None
    fid = None  # best ARI among 3-class columns
    opt_cache = {}
    for tag, meta, load in ctx.objects("pred"):
        if meta.get("kind") not in ("gdf", "df") or abs(meta.get("rows", 0) - N) > 0.02 * N:
            continue
        try:
            obj = load()
        except Exception:
            continue
        keys = key_values(obj, inputs, "ID")
        if keys is None:
            continue
        for c in obj.columns:
            if c in INPUT_COLS or str(c).lower() in ("geometry", "id", "name", "cacode", "ename", "cname"):
                continue
            v = obj[c]
            if v.dtype.kind in "OSU" or str(v.dtype) in ("category", "string"):
                # text / categorical cluster labels ("Profile A")
                if v.isna().any() or not 2 <= v.nunique() <= 10:
                    continue
                v = pd.Series(pd.factorize(v.astype(str))[0], index=v.index)
            elif v.dtype.kind not in "iuf":
                continue
            if v.isna().any() or not np.allclose(v, np.round(v)):
                continue
            k = v.nunique()
            if not 2 <= k <= 10:
                continue
            lab = v.groupby(keys.values).first().reindex(gold.index)
            if lab.isna().mean() > 0.05:
                continue
            m = lab.notna().to_numpy()
            labs = lab[m].to_numpy()
            if k not in opt_cache:
                opt_cache[k] = max(_r2(KMeans(k, n_init=10, random_state=s).fit_predict(Z), Z) for s in range(2))
            eff = _r2(labs, Z[m]) / opt_cache[k]
            ari = adjusted_rand_score(gold_lab[m], labs)
            if best is None or eff > best[0]:
                best = (eff, f"{c} in {meta['name']}@{tag}", k, ari)
            if k == 3 and (fid is None or ari > fid[0]):
                fid = (ari, f"{c} in {meta['name']}@{tag}")
    if best is None:
        out.append(result("clustering is a near-optimal K-means partition of the four risk scores", False,
                          note="no cluster-label column (2-10 classes) per constituency"))
        out.append(result("method fidelity: SEEDS-initialised K-means (3 clusters, ARI>=0.8 with gold)", False,
                          note="instructed: K-means initialised from SEEDS"))
    else:
        eff, where, k, ari = best
        out.append(result("clustering is a near-optimal K-means partition of the four risk scores", eff >= 0.90,
                          value=round(eff, 3), threshold=0.90, note=f"{where}, k={k}"))
        # required; best ARI over all 3-class cluster columns
        out.append(result("method fidelity: SEEDS-initialised K-means (3 clusters, ARI>=0.8 with gold)",
                          fid is not None and fid[0] >= 0.8, value=round(fid[0], 3) if fid else None,
                          threshold=0.8, note=(fid[1] if fid else f"no 3-class column (best k={k})")
                          + "; instructed: K-means initialised from SEEDS"))
    return out
