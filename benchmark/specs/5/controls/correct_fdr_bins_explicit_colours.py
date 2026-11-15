# Correct: 500 m pointy-top
# hexagons in UTM 10N, Gi* (distance band 2.5 km, 999 permutations), FDR-corrected 90/95/99 %
# hot/cold bins. The hexagons are clipped to the ZIP area (some become multi-part) and the bins
# are drawn with one explicit colour per bin (color=Series.map(...)), next to a count panel.
import os
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt, matplotlib.patches as mpatches
from shapely.geometry import Polygon
from libpysal.weights import DistanceBand
from esda.getisord import G_Local
from scipy.stats import norm

pts = gpd.read_file('dataset/High_Blood_Level_Results.shp')
zips = gpd.read_file('dataset/Sacramento_ZIP_Codes.shp')
crs = 'EPSG:32610'
p = pts.to_crs(crs); z = zips.to_crs(crs)
study = z.union_all()
xy = np.c_[p.geometry.x, p.geometry.y]
s = 500.0; w = np.sqrt(3) * s; h = 1.5 * s
minx, miny, maxx, maxy = np.array(z.total_bounds)
minx = min(minx, xy[:, 0].min()) - w; miny = min(miny, xy[:, 1].min()) - s
maxx = max(maxx, xy[:, 0].max()) + w; maxy = max(maxy, xy[:, 1].max()) + s
hexes = []; r = 0; y = miny
while y < maxy + s:
    x = minx + (w / 2 if r % 2 else 0)
    while x < maxx + w:
        ang = np.deg2rad(np.arange(30, 390, 60))
        hexes.append(Polygon(np.c_[x + s * np.cos(ang), y + s * np.sin(ang)]))
        x += w
    y += h; r += 1
grid = gpd.GeoDataFrame(geometry=hexes, crs=crs)
j = gpd.sjoin(p, grid, predicate='within', how='left')
grid['count'] = j.groupby('index_right').size().reindex(grid.index).fillna(0).astype(int)
keep = grid.intersects(study) | (grid['count'] > 0)
hexg = grid[keep].reset_index(drop=True)

cent = np.c_[hexg.centroid.x, hexg.centroid.y]
yv = hexg['count'].values.astype(float)
W = DistanceBand(cent, threshold=2500, binary=True, silence_warnings=True)
g = G_Local(yv, W, transform='B', star=True, permutations=999, seed=0)
z_ = g.Zs; pv = 2 * norm.sf(np.abs(z_))


def fdr_thresh(p_, a):
    ps = np.sort(p_); n = len(p_); ok = ps <= a * np.arange(1, n + 1) / n
    return ps[ok].max() if ok.any() else 0


bins = np.zeros(len(z_), int)
for conf, a in [(1, 0.10), (2, 0.05), (3, 0.01)]:
    t = fdr_thresh(pv, a); sig = pv <= t
    bins[sig] = np.sign(z_[sig]) * conf
hexg['GiZ'] = z_; hexg['GiP'] = pv; hexg['Gi_Bin'] = bins

os.makedirs('pred_results', exist_ok=True)
hexc = hexg.copy()
hexc['geometry'] = hexg.intersection(study.buffer(0)).where(hexg.intersects(study), hexg.geometry)
cols = {-3: '#4575b4', -2: '#91bfdb', -1: '#e0f3f8', 0: '#f0f0f0', 1: '#fee090', 2: '#fc8d59', 3: '#d73027'}
labs = {-3: 'Cold Spot - 99%', -2: 'Cold Spot - 95%', -1: 'Cold Spot - 90%', 0: 'Not Significant',
        1: 'Hot Spot - 90%', 2: 'Hot Spot - 95%', 3: 'Hot Spot - 99%'}
fig, ax = plt.subplots(1, 2, figsize=(18, 10))
hexc.plot(ax=ax[0], color=hexc.Gi_Bin.map(cols), edgecolor='white', linewidth=0.3)
z.boundary.plot(ax=ax[0], color='k', linewidth=0.8)
ax[0].legend(handles=[mpatches.Patch(color=cols[b], label=labs[b]) for b in sorted(hexc.Gi_Bin.unique())],
             loc='lower left', title='Gi* bin (FDR-corrected)')
hc = hexc[hexc['count'] > 0]
hexc.plot(ax=ax[1], color='#f7f7f7', edgecolor='#dddddd', linewidth=0.3)
hc.plot(ax=ax[1], column='count', cmap='YlOrRd', scheme='NaturalBreaks', k=5, edgecolor='grey', linewidth=0.3,
        legend=True, legend_kwds={'title': 'Cases per hexagon', 'loc': 'lower left'})
z.boundary.plot(ax=ax[1], color='k', linewidth=0.8)
hexc[hexc.Gi_Bin >= 2].dissolve().boundary.plot(ax=ax[1], color='blue', linewidth=1.5)
for a in ax:
    a.set_axis_off()
plt.tight_layout(); plt.savefig('pred_results/blood_lead_hotspots.png', dpi=120); plt.close()
hexc.to_file('pred_results/blood_lead_hex_hotspots.gpkg', driver='GPKG')
