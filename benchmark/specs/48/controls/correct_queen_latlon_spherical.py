# Correct: Gi* on Queen contiguity (row-standardised), GWR bisquare 22 NN on lat/lon with spherical distances.
import geopandas as gpd, numpy as np, libpysal, esda
from mgwr.gwr import GWR
d = gpd.read_file('dataset/loan_data.shp')
sel = d.query('AcceptedLo >= 30').reset_index(drop=True)
w = libpysal.weights.Queen.from_dataframe(sel, use_index=False)
gi = esda.G_Local(sel['AveInteres'].values, w, transform='R', star=True, permutations=0)
ll = sel.to_crs(4326).centroid
coords = np.c_[ll.x, ll.y]
res = GWR(coords, sel['AveInteres'].values.reshape(-1, 1), sel[['AveLoanGra']].values, bw=22, kernel='bisquare', spherical=True).fit()
out = sel.assign(hot_z=gi.Zs, coef_grade=res.params[:, 1], intercept=res.params[:, 0])
out.to_file('pred_results/gwr_result.shp')
