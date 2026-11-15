# Correct: ~2.6 km2 hexagons clipped to the ZIP-code
# union (edge cells are no longer six-vertex polygons, fewer than half of all cells are), counts
# per clipped hexagon, analytic Gi* (distance band ~2 spacings) on them, z mapped with column=.
import numpy as np, geopandas as gpd, libpysal, matplotlib.pyplot as plt
from shapely.geometry import Polygon
from esda.getisord import G_Local

pts = gpd.read_file('dataset/High_Blood_Level_Results.shp').to_crs(32610)
zips = gpd.read_file('dataset/Sacramento_ZIP_Codes.shp').to_crs(32610)
area = zips.union_all()
s = np.sqrt(2 * 2.6e6 / (3 * np.sqrt(3)))
xmin, ymin, xmax, ymax = zips.total_bounds
polys = []
for c, x in enumerate(np.arange(xmin - s, xmax + 2 * s, 1.5 * s)):
    for y in np.arange(ymin - s - (np.sqrt(3) * s / 2 if c % 2 else 0), ymax + 2 * s, np.sqrt(3) * s):
        polys.append(Polygon([(x + s * np.cos(np.radians(a)), y + s * np.sin(np.radians(a))) for a in range(0, 360, 60)]))
hx = gpd.GeoDataFrame(geometry=polys, crs=32610)
hx = hx[hx.intersects(area)].copy()
hx['geometry'] = hx.geometry.intersection(area)
hx = hx[hx.geom_type == 'Polygon'].reset_index(drop=True)
hx['n_cases'] = gpd.sjoin(pts, hx, predicate='within').groupby('index_right').size().reindex(hx.index, fill_value=0)
cent = np.c_[hx.centroid.x, hx.centroid.y]
w = libpysal.weights.DistanceBand(cent, threshold=2.1 * np.sqrt(3) * s, binary=True, silence_warnings=True)
hx['gi_z'] = G_Local(hx['n_cases'].values.astype(float), w, star=True, permutations=0).Zs
ax = hx.plot(column='gi_z', cmap='RdBu_r', vmin=-4, vmax=4, legend=True, edgecolor='white', linewidth=0.3, figsize=(8, 10))
zips.boundary.plot(ax=ax, color='grey', linewidth=0.5)
plt.savefig('pred_results/blood_lead_hotspots.png')
