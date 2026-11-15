# Wrong: skips the peak-time selection; all crashes (all days and hours) are analysed.
import geopandas as gpd, numpy as np, libpysal, esda, matplotlib.pyplot as plt
c = gpd.read_file('dataset/crashes.shp'); r = gpd.read_file('dataset/roads.shp')
j = gpd.sjoin_nearest(c[['geometry']], r[['UniqID', 'geometry']], max_distance=1320)
j = j[~j.index.duplicated()]
r['rate'] = r.UniqID.map(j.UniqID.value_counts()).fillna(0) / (6 * r.Shape_Leng)
w = libpysal.io.open('dataset/nwswm360ft.swm', 'r').read()
w = libpysal.weights.W(w.neighbors, w.weights, id_order=r.UniqID.tolist(), silence_warnings=True)
g = esda.G_Local(r['rate'].values, w, star=True, permutations=0)
r['z'] = g.Zs
r.plot(column='z', cmap='RdBu_r', linewidth=0.5); plt.savefig('pred_results/crash_hotspots.png')
