# Wrong: the cost surface lacks the road layer and the "connections" are straight lines between habitat
# centroids instead of least-cost paths.
import numpy as np, rasterio, geopandas as gpd, matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, minimum_filter
from shapely.geometry import LineString
src = rasterio.open('dataset/Elevation.tif'); elev = src.read(1).astype(float)
lc = rasterio.open('dataset/landCover.tif').read(1); ps = rasterio.open('dataset/Protected_Status.tif').read(1)
hab = gpd.read_file('dataset/habitats.geojson')
rug = maximum_filter(elev, 3) - minimum_filter(elev, 3)
lut = np.full(256, np.nan); lut[[11, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95]] = [10, 8, 7, 8, 9, 6, 2, 1, 2, 3, 3, 4, 6, 4, 1]
pc = np.full(ps.shape, 9.0); pc[ps == 1] = 1; pc[ps == 2] = 3; pc[ps == 3] = 6
cost = 1.25 * (10 - 9 * (rug - rug.min()) / (rug.max() - rug.min())) + lut[lc] + pc
# "connections": straight lines from the central habitat to the others (cost surface not used)
c = hab.geometry.centroid
links = gpd.GeoDataFrame(geometry=[LineString([c[0].coords[0], c[k].coords[0]]) for k in (1, 2, 3)], crs=hab.crs)
b = src.bounds
fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(cost, extent=[b.left, b.right, b.bottom, b.top]); links.plot(ax=ax, color='r'); hab.boundary.plot(ax=ax, color='w')
plt.savefig('pred_results/optimal_path.png')
