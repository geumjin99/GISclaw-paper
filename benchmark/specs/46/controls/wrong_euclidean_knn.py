# Wrong: ignores the network SWM; Gi* with 8 nearest Euclidean neighbours of segment centroids.
import geopandas as gpd, numpy as np, libpysal, esda, matplotlib.pyplot as plt
c = gpd.read_file('dataset/crashes.shp'); r = gpd.read_file('dataset/roads.shp')
sel = c[c.CRASH_DOW.isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']) & c.CRASH_HOD.between(15, 17)]
j = gpd.sjoin_nearest(sel[['geometry']], r[['UniqID', 'geometry']], max_distance=1320)
j = j[~j.index.duplicated()]
r['rate'] = r.UniqID.map(j.UniqID.value_counts()).fillna(0) / (6 * r.Shape_Leng)
w = libpysal.weights.KNN.from_dataframe(r.set_geometry(r.centroid), k=8)
g = esda.G_Local(r['rate'].values, w, star=True, transform='B', permutations=0)
r['z'] = g.Zs
r.plot(column='z', cmap='RdBu_r', linewidth=0.5); plt.savefig('pred_results/crash_hotspots.png')
