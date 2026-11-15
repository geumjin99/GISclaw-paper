# Correct: linear rescaling (not MSLarge), unknown protection -> 5, centroid-to-centroid
# route_through_array + MST, paths drawn in pixel coordinates.
import numpy as np, rasterio, geopandas as gpd, matplotlib.pyplot as plt, itertools, networkx as nx
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt, maximum_filter, minimum_filter
from skimage.graph import route_through_array
src = rasterio.open('dataset/Elevation.tif'); elev = src.read(1).astype(float)
lc = rasterio.open('dataset/landCover.tif').read(1); ps = rasterio.open('dataset/Protected_Status.tif').read(1)
hab = gpd.read_file('dataset/habitats.geojson')
roads = gpd.read_file('dataset/Roads.geojson').to_crs(src.crs)
rug = maximum_filter(elev, 3) - minimum_filter(elev, 3)
rug_cost = 10 - 9 * (rug - rug.min()) / (rug.max() - rug.min())          # linear rescale instead of MSLarge
on_road = rasterize(((g, 1) for g in roads.geometry), out_shape=elev.shape, transform=src.transform, fill=0, dtype='uint8')
dist = distance_transform_edt(on_road == 0) * 30.0
road_cost = 10 - 9 * np.clip(dist / 3000.0, 0, 1)                         # linear, saturating at 3 km
lut = np.full(256, np.nan); lut[[11, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95]] = [10, 8, 7, 8, 9, 6, 2, 1, 2, 3, 3, 4, 6, 4, 1]
lc_cost = lut[lc]
pcost = np.full(ps.shape, 5.0); pcost[ps == 1] = 1; pcost[ps == 2] = 3; pcost[ps == 3] = 6; pcost[ps == 4] = 9   # unknown status -> 5
cost = 1.25 * rug_cost + road_cost + lc_cost + pcost
cost_for_routing = np.where(np.isfinite(cost), cost, 1e6)
cent = [src.index(p.x, p.y) for p in hab.representative_point()]
G = nx.Graph(); routes = {}
for i, j in itertools.combinations(range(len(cent)), 2):
    path, c = route_through_array(cost_for_routing, cent[i], cent[j], fully_connected=True, geometric=True)
    G.add_edge(i, j, weight=c); routes[(i, j)] = np.array(path)
fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(cost, cmap='viridis_r')
for i, j in nx.minimum_spanning_tree(G).edges():
    p = routes[(min(i, j), max(i, j))]
    ax.plot(p[:, 1], p[:, 0], 'r-', lw=2)
plt.savefig('pred_results/optimal_path.png')
