# Wrong: Gi* with Euclidean inverse-distance weights and self weight 1 instead of the supplied
# network spatial weights.
import os, geopandas as gpd, numpy as np, scipy.sparse as sp, matplotlib.pyplot as plt
os.makedirs("pred_results", exist_ok=True)
crashes = gpd.read_file("dataset/crashes.shp"); roads = gpd.read_file("dataset/roads.shp")
peak = crashes[crashes.CRASH_DOW.isin(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]) & crashes.CRASH_HOD.between(15, 17)].copy()
snap = gpd.sjoin_nearest(peak[["geometry"]], roads[["UniqID", "geometry"]], max_distance=1320)
snap = snap[~snap.index.duplicated()]
roads["crashes"] = roads.UniqID.map(snap.UniqID.value_counts()).fillna(0)
roads["CrashRate"] = roads.crashes / (6 * roads.Shape_Leng)
x = roads.CrashRate.to_numpy()
# Gi* with EUCLIDEAN inverse-distance weights (segment midpoints within 1320 ft), self weight 1 (ArcGIS default)
from scipy.spatial import cKDTree
pts = np.c_[roads.geometry.centroid.x, roads.geometry.centroid.y]
D = cKDTree(pts).sparse_distance_matrix(cKDTree(pts), 1320.0, output_type="coo_matrix")
m = D.data > 0
n = len(x)
W = sp.csr_matrix((1.0 / D.data[m], (D.row[m], D.col[m])), shape=(n, n)) + sp.identity(n)
s1 = np.asarray(W.sum(1)).ravel(); s2 = np.asarray(W.multiply(W).sum(1)).ravel()
roads["GiZScore"] = (W @ x - x.mean() * s1) / (x.std() * np.sqrt((n * s2 - s1 ** 2) / (n - 1)))
roads.plot(column="GiZScore", cmap="RdBu_r", vmin=-3, vmax=3, linewidth=0.4, figsize=(8, 10))
plt.savefig("pred_results/crash_hotspots.png")
