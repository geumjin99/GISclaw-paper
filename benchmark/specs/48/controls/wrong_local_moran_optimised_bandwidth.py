# Wrong: Local Moran's I values stored as 'gi_zscore' instead of Gi*, and a GWR with an optimised
# bandwidth instead of the 22 neighbours the instruction asks for.
import geopandas as gpd, numpy as np, libpysal, esda
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
loan = gpd.read_file('dataset/loan_data.shp')
loan_sel = loan[loan['AcceptedLo'] >= 30].copy()
wq = libpysal.weights.KNN.from_dataframe(loan_sel, k=8); wq.transform = 'r'
lisa = esda.moran.Moran_Local(loan_sel['AveInteres'].values, wq, seed=1)
loan_sel['gi_zscore'] = lisa.Is
coords = list(zip(loan_sel.centroid.x, loan_sel.centroid.y))
y = loan_sel['AveInteres'].values.reshape(-1, 1); X = loan_sel[['AveLoanGra']].values
bw = Sel_BW(coords, y, X).search()
r = GWR(coords, y, X, bw=bw).fit()
loan_sel['gwr_coef'] = r.params[:, 1]
loan_sel.to_file('pred_results/gwr_result.shp')
