"""Correct: standardised variables; GWR at the Sel_BW optimum is compared with MGWR (per-variable
bandwidths) in a 2x4 figure of coefficient maps.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, calculate_default_transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')
os.makedirs('pred_results', exist_ok=True)


import pandas as pd, geopandas as gpd, numpy as np
gdata = pd.read_csv('dataset/GData_utm.csv')
gshp = gpd.read_file('dataset/G_utm.shp')
print(gdata.shape, gshp.shape, gshp.crs)
print(gdata[['AreaKey','PctBach','PctFB','PctBlack','PctRural','X','Y']].describe().T)
print(len(set(gdata.AreaKey)&set(gshp.AreaKey)))

from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
g_y = gdata['PctBach'].values.reshape(-1,1)
g_X = gdata[['PctFB','PctBlack','PctRural']].values
coords = list(zip(gdata.X, gdata.Y))
g_X = (g_X - g_X.mean(0))/g_X.std(0)
g_y = (g_y - g_y.mean(0))/g_y.std(0)
gwr_selector = Sel_BW(coords, g_y, g_X)
gwr_bw = gwr_selector.search()
print('GWR bw:', gwr_bw)
gwr_results = GWR(coords, g_y, g_X, gwr_bw).fit()
print('GWR AICc', gwr_results.aicc, 'R2', gwr_results.R2)
mgwr_selector = Sel_BW(coords, g_y, g_X, multi=True)
mgwr_bw = mgwr_selector.search()
print('MGWR bws:', mgwr_bw)
mgwr_results = MGWR(coords, g_y, g_X, mgwr_selector).fit()
print('MGWR AICc', mgwr_results.aicc, 'R2', mgwr_results.R2)

import matplotlib.pyplot as plt
names = ['Intercept','PctFB','PctBlack','PctRural']
coef = pd.DataFrame({'AreaKey': gdata.AreaKey})
for i,n in enumerate(names):
    coef['gwr_'+n] = gwr_results.params[:,i]; coef['mgwr_'+n] = mgwr_results.params[:,i]
gmap = gshp.merge(coef, on='AreaKey')
print(len(gmap)); print(coef.describe().T[['min','max']])
fig, axes = plt.subplots(2,4, figsize=(20,11))
for j,n in enumerate(names):
    vmin = min(gmap['gwr_'+n].min(), gmap['mgwr_'+n].min()); vmax = max(gmap['gwr_'+n].max(), gmap['mgwr_'+n].max())
    for i,(m,bw) in enumerate([('gwr', gwr_bw), ('mgwr', mgwr_bw[j])]):
        ax = axes[i,j]
        gmap.plot(column=f'{m}_{n}', cmap='viridis', vmin=vmin, vmax=vmax, ax=ax, edgecolor='grey', linewidth=0.3)
        ax.set_title(f'{m.upper()} {n} (bw={int(bw)})', fontsize=13); ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin, vmax))
    fig.colorbar(sm, ax=axes[:,j], orientation='horizontal', fraction=0.04, pad=0.03)
fig.suptitle(f'Georgia PctBach: GWR (bw={int(gwr_bw)}, AICc={gwr_results.aicc:.1f}) vs MGWR (bws={mgwr_bw.astype(int).tolist()}, AICc={mgwr_results.aicc:.1f})', fontsize=15)
plt.savefig('pred_results/GA_GWR.png', dpi=150, bbox_inches='tight'); plt.close()
import os; print(os.listdir('pred_results'))