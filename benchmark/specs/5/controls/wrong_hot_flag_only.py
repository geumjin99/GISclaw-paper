# Wrong: no hot spot analysis (no Gi*, no significance); a 0/1 "hot" flag (top 20 % of the case sum within
# ~2 hexagon spacings) is the only result, drawn one class at a time. The flag rank-agrees with Gi* z
# (rho ~0.68) but is not a set of confidence bins.
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from shapely.geometry import Polygon
from scipy.spatial import cKDTree

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
nb = cKDTree(cent).query_ball_point(cent, 2.1 * np.sqrt(3) * s)
local = np.array([hx['count'].values[n].sum() for n in nb])
hx['hot'] = (local >= np.percentile(local, 80)).astype(int)
fig, ax = plt.subplots(figsize=(8, 10))
for k, col in {0: '#eeeeee', 1: '#d7301f'}.items():
    hx[hx.hot == k].plot(ax=ax, color=col, edgecolor='white', linewidth=0.3)
plt.savefig('pred_results/blood_lead_hotspots.png')
