"""Reference solution for task 46: peak-hour crash hot spots on the Brevard County road network.

Follows the original ArcPy model:
  1. select weekday (Mon-Fri) crashes with CRASH_HOD between 15 and 17 (SQL >= 15 AND <= 17);
  2. copy the selection;
  3. snap the crash points to the nearest road edge within 0.25 mile (1 320 ft; the data are
     in NAD83(HARN) / Florida East, US feet);
  4. spatial join roads (target, KEEP_ALL) <- snapped crashes = crash count per 100 m segment;
  5. CrashRate = Join_Count / (6 years * segment length in feet);
  6. Getis-Ord Gi* on CrashRate with the supplied network spatial weights (nwswm360ft.swm,
     inverse network distance between segment midpoints in feet, keyed by UniqID), no
     standardisation, analytical z/p, FDR-corrected confidence bins (Gi_Bin -3..3) as in ArcGIS.
     Self-potential: each segment is weighted like its nearest network neighbour (row maximum).
     A self weight of 1 (ArcGIS/esda default for Gi*) is ~300x larger than the inverse-distance
     neighbour weights (~0.003) and turns Gi* into the standardised rate itself (r = 0.9999),
     i.e. a non-spatial ranking; that convention is not used here.
Outputs: pred_results/selected_crash_hotspots.shp, pred_results/crash_hotspots.png
"""
import numpy as np
import geopandas as gpd
import libpysal
import scipy.sparse as sp
from scipy.stats import norm
import matplotlib.pyplot as plt

crashes = gpd.read_file("dataset/crashes.shp")
roads = gpd.read_file("dataset/roads.shp")

# 1-2 peak selection (weekday 3-5 pm) and copy
weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
peak = crashes[crashes["CRASH_DOW"].isin(weekdays) & (crashes["CRASH_HOD"] >= 15) & (crashes["CRASH_HOD"] <= 17)].copy()
print("peak crashes:", len(peak), "of", len(crashes))

# 3-4 snap within 0.25 mile to the nearest road edge and count per segment
snapped = gpd.sjoin_nearest(peak[["CrashID", "geometry"]], roads[["UniqID", "geometry"]],
                            max_distance=1320.0, distance_col="snap_dist_ft")
snapped = snapped[~snapped.index.duplicated(keep="first")]
print("snapped:", len(snapped), "median snap distance ft:", round(float(snapped.snap_dist_ft.median()), 1))
counts = snapped.groupby("UniqID").size()
roads["Join_Count"] = roads["UniqID"].map(counts).fillna(0).astype(int)

# 5 crash rate (crashes per foot-year, as in the model; segment lengths in feet)
roads["CrashRate"] = roads["Join_Count"] / (6.0 * roads["Shape_Leng"])

# 6 Gi* with the network spatial weights matrix
w = libpysal.io.open("dataset/nwswm360ft.swm", "r").read()
pos = {u: i for i, u in enumerate(roads["UniqID"].tolist())}
n = len(roads)
r_i, c_i, v_i = [], [], []
for k, nbrs in w.neighbors.items():
    for nb, wt in zip(nbrs, w.weights[k]):
        r_i.append(pos[k]); c_i.append(pos[nb]); v_i.append(wt)
W = sp.csr_matrix((v_i, (r_i, c_i)), shape=(n, n))
row_max = np.asarray(W.max(axis=1).todense()).ravel()
self_w = np.where(row_max > 0, row_max, 1.0)          # isolated segments: self only
W = W + sp.diags(self_w)
x = roads["CrashRate"].to_numpy(float)
xbar, S = x.mean(), x.std()
sw = np.asarray(W.sum(axis=1)).ravel()
sw2 = np.asarray(W.multiply(W).sum(axis=1)).ravel()
gi_z = (W @ x - xbar * sw) / (S * np.sqrt((n * sw2 - sw ** 2) / (n - 1)))
gi_p = 2 * norm.sf(np.abs(gi_z))
roads["GiZScore"] = gi_z
roads["GiPValue"] = gi_p


def fdr_bins(z, p):
    """ArcGIS-style confidence bins with Benjamini-Hochberg FDR at 90/95/99 %."""
    order = np.sort(p)
    m = len(p)
    out = np.zeros(m, int)
    for alpha, level in ((0.10, 1), (0.05, 2), (0.01, 3)):
        crit = alpha * np.arange(1, m + 1) / m
        ok = np.nonzero(order <= crit)[0]
        thr = order[ok.max()] if len(ok) else -1.0
        sig = p <= thr
        out[sig] = level
    return np.sign(z).astype(int) * out


roads["Gi_Bin"] = fdr_bins(gi_z, gi_p)
print("Gi_Bin counts:", roads["Gi_Bin"].value_counts().sort_index().to_dict())
roads.to_file("pred_results/selected_crash_hotspots.shp")

colors = {-3: "#2166ac", -2: "#67a9cf", -1: "#d1e5f0", 0: "#e0e0e0", 1: "#fddbc7", 2: "#ef8a62", 3: "#b2182b"}
fig, ax = plt.subplots(figsize=(9, 12))
for b in sorted(colors):
    sub = roads[roads.Gi_Bin == b]
    if len(sub):
        sub.plot(ax=ax, color=colors[b], linewidth=0.4 if b == 0 else 1.2, label=f"Gi_Bin {b} ({len(sub)})")
ax.legend(loc="lower left", fontsize=8)
ax.set_title("Peak-hour (weekday 3-5 pm) crash-rate hot spots, Brevard County 2010-2015\nGetis-Ord Gi*, network weights, FDR")
ax.set_axis_off()
plt.savefig("pred_results/crash_hotspots.png", dpi=120, bbox_inches="tight")
