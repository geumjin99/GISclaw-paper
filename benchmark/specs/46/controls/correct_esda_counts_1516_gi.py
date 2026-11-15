# Correct: hours 15-16, counts (not rates), esda G_Local (Gi, star=False, row-standardised) on the supplied network SWM.
import geopandas as gpd, numpy as np, libpysal, esda, matplotlib.pyplot as plt
crashes = gpd.read_file('dataset/crashes.shp')
roads = gpd.read_file('dataset/roads.shp')
peak = crashes[crashes.CRASH_DOW.isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']) & crashes.CRASH_HOD.isin([15, 16])].copy()
j = gpd.sjoin_nearest(peak, roads[['UniqID', 'geometry']], max_distance=1320)
j = j[~j.index.duplicated()]
roads['n_crash'] = roads.UniqID.map(j.groupby('UniqID').size()).fillna(0)
w = libpysal.io.open('dataset/nwswm360ft.swm', 'r').read()
w = libpysal.weights.W(w.neighbors, w.weights, id_order=roads.UniqID.tolist(), silence_warnings=True)
g = esda.G_Local(roads['n_crash'].values, w, star=False, transform="R", permutations=0)
roads['z'] = g.Zs
ax = roads.plot(column='z', cmap='RdBu_r', linewidth=0.5, figsize=(8, 10), legend=True)
plt.savefig('pred_results/hotspots.png')
