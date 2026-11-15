# Correct: pointy-top 0.5 km2 hexagons in California Albers, Gi* with k-nearest-neighbour
# weights (k = 8), map of the Gi* z-scores and of the counts.
import numpy as np, geopandas as gpd, libpysal, matplotlib.pyplot as plt
from shapely.geometry import Polygon
from esda.getisord import G_Local

pts = gpd.read_file('dataset/High_Blood_Level_Results.shp').to_crs(3310)
zips = gpd.read_file('dataset/Sacramento_ZIP_Codes.shp').to_crs(3310)
r = np.sqrt(2 * 0.5e6 / (3 * np.sqrt(3)))
dx, dy = np.sqrt(3) * r, 1.5 * r
x0, y0, x1, y1 = zips.total_bounds
cells = []
for i, y in enumerate(np.arange(y0 - r, y1 + r, dy)):
    for x in np.arange(x0 - r + (dx / 2 if i % 2 else 0), x1 + r, dx):
        cells.append(Polygon([(x + r * np.cos(np.radians(a)), y + r * np.sin(np.radians(a))) for a in range(30, 390, 60)]))
hexgrid = gpd.GeoDataFrame(geometry=cells, crs=3310)
hexgrid = hexgrid[hexgrid.intersects(zips.union_all())].reset_index(drop=True)
hexgrid['n_cases'] = gpd.sjoin(pts, hexgrid, predicate='within').groupby('index_right').size().reindex(hexgrid.index, fill_value=0)
w = libpysal.weights.KNN.from_dataframe(hexgrid, k=8)
g = G_Local(hexgrid['n_cases'].values.astype(float), w, star=True, permutations=199, seed=1, n_jobs=1)
hexgrid['z'] = g.Zs
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
hexgrid.plot(column='z', cmap='RdBu_r', ax=ax[0], legend=True)
hexgrid.plot(column='n_cases', cmap='Reds', ax=ax[1], legend=True)
plt.savefig('pred_results/transit_access.png')
