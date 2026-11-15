# Wrong: Gi* with an explicit self weight of 1 on the inverse-distance network SWM
# (weights ~0.003) - the ArcGIS-style convention. Gi* collapses to the standardised rate (r 0.9999):
# no spatial hot-spot information. Expected: Gi* check UNVERIFIABLE -> not correct.
import geopandas as gpd, numpy as np, libpysal, esda, matplotlib.pyplot as plt
crashes = gpd.read_file('dataset/crashes.shp')
roads = gpd.read_file('dataset/roads.shp')
peak = crashes[crashes.CRASH_DOW.isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']) & crashes.CRASH_HOD.between(15, 17)].copy()
j = gpd.sjoin_nearest(peak, roads[['UniqID', 'geometry']], max_distance=1320)
j = j[~j.index.duplicated()]
roads['n_crash'] = roads.UniqID.map(j.groupby('UniqID').size()).fillna(0)
roads['rate'] = roads['n_crash'] / (6 * roads['Shape_Leng'])
w = libpysal.io.open('dataset/nwswm360ft.swm', 'r').read()
w = libpysal.weights.W(w.neighbors, w.weights, id_order=roads.UniqID.tolist(), silence_warnings=True)
g = esda.G_Local(roads['rate'].values, w, star=1.0, transform='R', permutations=0)
roads['GiZScore'] = g.Zs
ax = roads.plot(column='GiZScore', cmap='RdBu_r', linewidth=0.5, figsize=(8, 10), legend=True)
plt.savefig('pred_results/hotspots.png')
