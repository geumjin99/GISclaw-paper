# Correct: Gi* with KNN-8 weights (esda), GWR with an adaptive Gaussian kernel of 22 neighbours.
import geopandas as gpd, numpy as np, libpysal, esda, matplotlib.pyplot as plt
from mgwr.gwr import GWR
d = gpd.read_file('dataset/loan_data.shp')
sel = d[d.AcceptedLo >= 30].copy()
w = libpysal.weights.KNN.from_dataframe(sel.set_geometry(sel.centroid), k=8)
g = esda.G_Local(sel.AveInteres.values, w, star=True, permutations=0)
sel['Gi_z'] = g.Zs
c = np.c_[sel.centroid.x, sel.centroid.y]
r = GWR(c, sel[['AveInteres']].values, sel[['AveLoanGra']].values, bw=22, fixed=False, kernel='gaussian').fit()
sel['b_grade'] = r.params[:, 1]; sel['localR2'] = r.localR2[:, 0]
sel.to_file('pred_results/loan_analysis.gpkg')
fig, ax = plt.subplots(1, 2); sel.plot('Gi_z', ax=ax[0]); sel.plot('b_grade', ax=ax[1]); plt.savefig('pred_results/out.png')
