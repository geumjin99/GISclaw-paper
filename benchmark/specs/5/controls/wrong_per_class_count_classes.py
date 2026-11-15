# Wrong: correct Gi* z on 2 km2 hexagons, but the map draws
# case-count classes one class at a time (single-colour collections), not the hot spot result.
import numpy as np, geopandas as gpd, libpysal, matplotlib.pyplot as plt
from shapely.geometry import Polygon
from esda.getisord import G_Local

pts = gpd.read_file('dataset/High_Blood_Level_Results.shp').to_crs(32610)
zips = gpd.read_file('dataset/Sacramento_ZIP_Codes.shp').to_crs(32610)
s = np.sqrt(2 * 2e6 / (3 * np.sqrt(3)))
xmin, ymin, xmax, ymax = zips.total_bounds
polys = []
for c, x in enumerate(np.arange(xmin, xmax + 2 * s, 1.5 * s)):
    for y in np.arange(ymin - (np.sqrt(3) * s / 2 if c % 2 else 0), ymax + 2 * s, np.sqrt(3) * s):
        polys.append(Polygon([(x + s * np.cos(np.radians(a)), y + s * np.sin(np.radians(a))) for a in range(0, 360, 60)]))
hx = gpd.GeoDataFrame(geometry=polys, crs=32610)
hx = hx[hx.intersects(zips.union_all())].reset_index(drop=True)
hx['count'] = gpd.sjoin(pts, hx, predicate='within').groupby('index_right').size().reindex(hx.index, fill_value=0)
cent = np.c_[hx.centroid.x, hx.centroid.y]
w = libpysal.weights.DistanceBand(cent, threshold=2.1 * np.sqrt(3) * s, binary=True, silence_warnings=True)
hx['gi_z'] = G_Local(hx['count'].values.astype(float), w, star=True, permutations=0).Zs
hx['cnt_class'] = np.select([hx['count'] == 0, hx['count'] <= 2, hx['count'] <= 5], [0, 1, 2], 3)
cols = {0: '#f7f7f7', 1: '#fdd49e', 2: '#fc8d59', 3: '#b30000'}
fig, ax = plt.subplots(figsize=(8, 10))
for k, col in cols.items():
    sub = hx[hx.cnt_class == k]
    if len(sub):
        sub.plot(ax=ax, color=col, edgecolor='white', linewidth=0.3)
plt.savefig('pred_results/blood_lead_hotspots.png')
