# Wrong: the Gi* analysis is run correctly (as in correct_fdr_bins_explicit_colours) but the map only
# shows case counts: the hexagons are coloured by count class with one explicit colour per class,
# so the hot spot result is never drawn.
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
cls = np.digitize(hexc['count'], [1, 2, 4, 8])
pal = ['#f7f7f7', '#fee5d9', '#fcae91', '#fb6a4a', '#cb181d']
fig, ax = plt.subplots(figsize=(10, 10))
hexc.plot(ax=ax, color=[pal[k] for k in cls], edgecolor='white', linewidth=0.3)
z.boundary.plot(ax=ax, color='k', linewidth=0.8)
ax.legend(handles=[mpatches.Patch(color=c, label=l) for c, l in zip(pal, ['0', '1', '2-3', '4-7', '8+'])],
          loc='lower left', title='Cases per hexagon')
ax.set_axis_off()
plt.tight_layout(); plt.savefig('pred_results/blood_lead_hotspots.png', dpi=120); plt.close()
hexc.to_file('pred_results/blood_lead_hex_hotspots.gpkg', driver='GPKG')
