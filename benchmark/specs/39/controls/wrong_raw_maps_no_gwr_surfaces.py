"""Wrong: searches a bandwidth but fits only a global OLS and maps the raw input attributes;
no local GWR coefficients are estimated or drawn.
"""
import geopandas as gpd, numpy as np, matplotlib.pyplot as plt
import statsmodels.api as sm
from mgwr.sel_bw import Sel_BW
g = gpd.read_file('dataset/G_utm.shp')
y = g['PctBach'].values.reshape(-1, 1)
X = g[['PctFB', 'PctBlack', 'PctRural']].values
bw = Sel_BW(list(zip(g.X, g.Y)), y, X).search()
ols = sm.OLS(y, sm.add_constant(X)).fit()        # only a global model is fitted
g['fitted'] = ols.fittedvalues
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for ax, c in zip(axes, ['PctBach', 'PctFB', 'PctBlack', 'PctRural']):
    g.plot(c, ax=ax, legend=True); ax.set_title(f'{c} (bw={bw:.0f})')
plt.savefig('pred_results/GA_GWR.png')
