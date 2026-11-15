"""Reference solution for task 48: interest-rate hot spots and GWR of LendingClub loans by ZIP3.

  1. select tracts (ZIP3 polygons) with AcceptedLo >= 30 and copy them (808 of 878);
  2. Hot Spot Analysis (Getis-Ord Gi*) on AveInteres: fixed distance band = ArcGIS default
     (smallest distance giving every feature >= 1 neighbour, centroid based: 202.6 km in the
     layer's USA Contiguous Albers metres), binary weights incl. self, row standardisation,
     analytical z, FDR-corrected bins;
  3. GWR AveInteres ~ AveLoanGra, adaptive kernel with 22 neighbours, bisquare weighting
     (as in the benchmark's ArcGIS model), on polygon centroids.
Outputs: pred_results/hot_spots_result.shp, pred_results/gwr_result.shp, pred_results/interest_rates.png
"""
import numpy as np
import geopandas as gpd
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import norm
from mgwr.gwr import GWR
import matplotlib.pyplot as plt

loans = gpd.read_file("dataset/loan_data.shp")
loan_selected = loans[loans["AcceptedLo"] >= 30].copy().reset_index(drop=True)      # select + copy
print("selected tracts:", len(loan_selected), "of", len(loans))

# --- Gi* hot spots
cent = np.c_[loan_selected.centroid.x, loan_selected.centroid.y]
tree = cKDTree(cent)
band = float(tree.query(cent, k=2)[0][:, 1].max())
print("distance band (m):", round(band, 1))
pairs = tree.query_pairs(band, output_type="ndarray")
n = len(loan_selected)
B = sp.coo_matrix((np.ones(2 * len(pairs)), (np.r_[pairs[:, 0], pairs[:, 1]], np.r_[pairs[:, 1], pairs[:, 0]])),
                  shape=(n, n)).tocsr() + sp.identity(n, format="csr")              # Gi*: self included
rs = np.asarray(B.sum(axis=1)).ravel()
W = sp.diags(1.0 / rs) @ B                                                          # row standardisation
x = loan_selected["AveInteres"].to_numpy(float)
s1 = np.asarray(W.sum(axis=1)).ravel(); s2 = np.asarray(W.multiply(W).sum(axis=1)).ravel()
gi_z = (W @ x - x.mean() * s1) / (x.std() * np.sqrt((n * s2 - s1 ** 2) / (n - 1)))
gi_p = 2 * norm.sf(np.abs(gi_z))


def fdr_bins(z, p):
    order = np.sort(p); m = len(p); out = np.zeros(m, int)
    for alpha, level in ((0.10, 1), (0.05, 2), (0.01, 3)):
        ok = np.nonzero(order <= alpha * np.arange(1, m + 1) / m)[0]
        thr = order[ok.max()] if len(ok) else -1.0
        out[p <= thr] = level
    return np.sign(z).astype(int) * out


hot_spots = loan_selected.copy()
hot_spots["GiZScore"] = gi_z
hot_spots["GiPValue"] = gi_p
hot_spots["Gi_Bin"] = fdr_bins(gi_z, gi_p)
hot_spots.to_file("pred_results/hot_spots_result.shp")
print("Gi_Bin counts:", hot_spots["Gi_Bin"].value_counts().sort_index().to_dict())

# --- GWR, 22 nearest neighbours, bisquare
y = x.reshape(-1, 1)
X = loan_selected[["AveLoanGra"]].to_numpy(float)
gwr_res = GWR(cent, y, X, bw=22, fixed=False, kernel="bisquare").fit()
gwr_result = loan_selected.copy()
gwr_result["Intercept"] = gwr_res.params[:, 0]
gwr_result["C1_AveLoan"] = gwr_res.params[:, 1]          # local coefficient of AveLoanGra
gwr_result["LocalR2"] = gwr_res.localR2.ravel()
gwr_result["Predicted"] = gwr_res.predy.ravel()
gwr_result["StdResid"] = gwr_res.std_res.ravel()
gwr_result.to_file("pred_results/gwr_result.shp")
print("GWR R2 %.3f AICc %.1f; local coef AveLoanGra %.3f..%.3f" % (gwr_res.R2, gwr_res.aicc,
      gwr_result.C1_AveLoan.min(), gwr_result.C1_AveLoan.max()))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
hot_spots.plot(column="Gi_Bin", cmap="RdBu_r", vmin=-3, vmax=3, legend=True, ax=axes[0], edgecolor="none")
axes[0].set_title("Interest-rate hot spots (Gi*, FDR bins)"); axes[0].set_axis_off()
gwr_result.plot(column="C1_AveLoan", cmap="viridis", legend=True, ax=axes[1], edgecolor="none")
axes[1].set_title("GWR local coefficient of average loan grade (22 NN, bisquare)"); axes[1].set_axis_off()
plt.tight_layout()
plt.savefig("pred_results/interest_rates.png", dpi=110)
