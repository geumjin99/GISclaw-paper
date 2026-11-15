"""Wrong: fits the GWR with PctPov as the dependent variable; the task models PctBach."""
import geopandas as gpd, numpy as np, matplotlib.pyplot as plt
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import GWR
g = gpd.read_file('dataset/G_utm.shp')
y = g['PctPov'].values.reshape(-1, 1)          # wrong dependent variable
X = g[['PctFB', 'PctBlack', 'PctRural']].values
coords = list(zip(g.X, g.Y))
bw = Sel_BW(coords, y, X).search()
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, b in zip(axes, [50, bw, 150]):
    params = GWR(coords, y, X, b).fit().params
    g['coef'] = params[:, 3]
    g.plot('coef', ax=ax, legend=True); ax.set_title(f'bw {b}')
plt.savefig('pred_results/GA_GWR.png')
