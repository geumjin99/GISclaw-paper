# Correct: Gi (no self) of AveInteres on row-standardised KNN-22 weights (the same 22 neighbours
# as the GWR); a legitimate weights choice whose z correlates about equally with rate and grade Gi*.
import geopandas as gpd, numpy as np, libpysal, esda, matplotlib.pyplot as plt
from mgwr.gwr import GWR
d = gpd.read_file('dataset/loan_data.shp')
sel = d[d.AcceptedLo >= 30].reset_index(drop=True)
c = np.c_[sel.centroid.x, sel.centroid.y]
w = libpysal.weights.KNN(c, k=22); w.transform = 'r'
sel['Gi_z'] = esda.G_Local(sel.AveInteres.values, w, star=False, permutations=0).Zs
r = GWR(c, sel[['AveInteres']].values, sel[['AveLoanGra']].values, bw=22, fixed=False, kernel='bisquare').fit()
sel['b_grade'] = r.params[:, 1]
sel.to_file('pred_results/loan_hotspots_gwr.shp')
fig, ax = plt.subplots(1, 2); sel.plot('Gi_z', ax=ax[0]); sel.plot('b_grade', ax=ax[1]); plt.savefig('pred_results/out.png')
