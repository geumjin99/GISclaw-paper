# Correct: SWM used as a binary network-neighbour structure (Gi* with self + network neighbours), rates per mile-year.
import geopandas as gpd, numpy as np, libpysal, scipy.sparse as sp, matplotlib.pyplot as plt
from scipy.stats import norm
c = gpd.read_file('dataset/crashes.shp'); r = gpd.read_file('dataset/roads.shp')
sel = c.query("CRASH_DOW in ['Monday','Tuesday','Wednesday','Thursday','Friday'] and 15 <= CRASH_HOD <= 17")
j = gpd.sjoin_nearest(sel[['geometry']], r[['UniqID', 'geometry']], max_distance=1320)
j = j[~j.index.duplicated()]
r['cnt'] = r.UniqID.map(j.UniqID.value_counts()).fillna(0)
r['rate_mi_yr'] = r.cnt / (6 * r.Shape_Leng / 5280)
w = libpysal.io.open('dataset/nwswm360ft.swm', 'r').read()
w.transform = 'B'
pos = {u: i for i, u in enumerate(r.UniqID)}
rows = [pos[k] for k in w.neighbors for _ in w.neighbors[k]]; cols = [pos[n] for k in w.neighbors for n in w.neighbors[k]]
n = len(r); M = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)) + sp.identity(n)
x = r.rate_mi_yr.values; s1 = np.asarray(M.sum(1)).ravel()
z = (M @ x - x.mean() * s1) / (x.std() * np.sqrt((n * s1 - s1 ** 2) / (n - 1)))
r['GiZ'] = z
r.plot(column='GiZ', cmap='RdBu_r', linewidth=0.5, figsize=(8, 10))
plt.savefig('pred_results/crash_hotspots.png')
