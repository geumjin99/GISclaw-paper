# Wrong: skips the AcceptedLo >= 30 selection (all 878 tracts); otherwise like the reference.
import geopandas as gpd, numpy as np, libpysal, esda
from mgwr.gwr import GWR
d = gpd.read_file('dataset/loan_data.shp')
w = libpysal.weights.KNN.from_dataframe(d.set_geometry(d.centroid), k=8)
d['Gi_z'] = esda.G_Local(d.AveInteres.values, w, star=True, permutations=0).Zs
c = np.c_[d.centroid.x, d.centroid.y]
d['coef'] = GWR(c, d[['AveInteres']].values, d[['AveLoanGra']].values, bw=22, fixed=False, kernel='bisquare').fit().params[:, 1]
d.to_file('pred_results/result.shp')
