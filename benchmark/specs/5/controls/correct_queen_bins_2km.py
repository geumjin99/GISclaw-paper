# Correct: 2 km2 flat-top hexagons (UTM 10N), Gi* with queen contiguity and 999 permutations,
# reported as 90/95/99 % hot/cold bins.
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
w = libpysal.weights.Queen.from_dataframe(hx, use_index=False)
gi = G_Local(hx['count'].values.astype(float), w, star=True, permutations=999, seed=0, n_jobs=1)
conf = np.select([gi.p_sim <= 0.01, gi.p_sim <= 0.05, gi.p_sim <= 0.10], [3, 2, 1], 0)
hx['hotspot_bin'] = np.sign(gi.Zs) * conf
ax = hx.plot(column='hotspot_bin', cmap='RdBu_r', categorical=True, legend=True, figsize=(8, 10))
plt.savefig('pred_results/transit_access.png')
