# Correct: Gi* 5-class diverging map (cold 99 / cold 95 / not significant / hot 95 / hot 99), drawn one class
# at a time.
import numpy as np, geopandas as gpd, libpysal, matplotlib.pyplot as plt
from shapely.geometry import Polygon
from esda.getisord import G_Local
from shapely.geometry import Polygon
from esda.getisord import G_Local

pts = gpd.read_file('dataset/High_Blood_Level_Results.shp').to_crs(32610)
zips = gpd.read_file('dataset/Sacramento_ZIP_Codes.shp').to_crs(32610)
s = 440.0
xmin, ymin, xmax, ymax = zips.total_bounds
polys = []
for c, x in enumerate(np.arange(xmin, xmax + 2 * s, 1.5 * s)):
    for y in np.arange(ymin - (np.sqrt(3) * s / 2 if c % 2 else 0), ymax + 2 * s, np.sqrt(3) * s):
        polys.append(Polygon([(x + s * np.cos(np.radians(a)), y + s * np.sin(np.radians(a))) for a in range(0, 360, 60)]))
hex_grid = gpd.GeoDataFrame(geometry=polys, crs=32610)
hex_grid = hex_grid[hex_grid.intersects(zips.union_all())].reset_index(drop=True)
hex_grid['case_count'] = gpd.sjoin(pts, hex_grid, predicate='within').groupby('index_right').size() \
    .reindex(hex_grid.index, fill_value=0)
cent = np.c_[hex_grid.centroid.x, hex_grid.centroid.y]
w = libpysal.weights.DistanceBand(cent, threshold=2.1 * np.sqrt(3) * s, binary=True, silence_warnings=True)
hex_grid['Gi_z'] = G_Local(hex_grid['case_count'].values.astype(float), w, star=True, permutations=0).Zs
hex_grid['hot_significant'] = hex_grid['Gi_z'] >= 1.645
cls = np.digitize(hex_grid.Gi_z, [-2.576, -1.96, 1.96, 2.576])
fig, ax = plt.subplots(figsize=(10, 10))
for k, col in zip(range(5), ['#2166ac', '#67a9cf', '#eeeeee', '#ef8a62', '#b2182b']):
    sub = hex_grid[cls == k]
    if len(sub): sub.plot(ax=ax, color=col, edgecolor='white', linewidth=0.2)
print('n sig hot', int((hex_grid.Gi_z>=1.645).sum()), 'n z>1', int((hex_grid.Gi_z>1).sum()))
plt.savefig('pred_results/blood_lead_hotspots.png')
