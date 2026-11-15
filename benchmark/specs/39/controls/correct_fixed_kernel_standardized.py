"""Correct: fixed bisquare kernel (bandwidth in metres) on standardised variables; the PctRural
coefficient is mapped for several bandwidths.
"""
import geopandas as gpd, numpy as np, matplotlib.pyplot as plt
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import GWR
g = gpd.read_file('dataset/G_utm.shp')
g['cenx'] = g.geometry.centroid.x; g['ceny'] = g.geometry.centroid.y
y = g[['PctBach']].values; X = g[['PctFB', 'PctBlack', 'PctRural']].values
y = (y - y.mean(0)) / y.std(0); X = (X - X.mean(0)) / X.std(0)
coords = np.c_[g.cenx, g.ceny]
best_bw = Sel_BW(coords, y, X, fixed=True, kernel='bisquare').search()
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for ax, bw in zip(axes, [best_bw * 0.5, best_bw, best_bw * 2, best_bw * 4]):
    params = GWR(coords, y, X, bw, fixed=True, kernel='bisquare').fit().params
    g['b_rural'] = params[:, 3]
    g.plot('b_rural', ax=ax, legend=True, cmap='viridis'); ax.set_title(f'fixed bw {bw/1000:.0f} km')
plt.savefig('pred_results/GA_GWR.png')
