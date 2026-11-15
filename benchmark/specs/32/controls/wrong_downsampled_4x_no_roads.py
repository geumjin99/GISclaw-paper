# Wrong: cost model on a 4x coarser grid (acceptable), but the distance-to-roads cost layer the task asks
# for is omitted.
import numpy as np, rasterio, geopandas as gpd, matplotlib.pyplot as plt, itertools, networkx as nx
from scipy.ndimage import maximum_filter, minimum_filter
from skimage.graph import route_through_array
src = rasterio.open("dataset/Elevation.tif"); elev = src.read(1).astype(float)
lc = rasterio.open("dataset/landCover.tif").read(1); ps = rasterio.open("dataset/Protected_Status.tif").read(1)
hab = gpd.read_file("dataset/habitats.geojson")
rug = maximum_filter(elev, 3) - minimum_filter(elev, 3)
rug_cost = 10 - 9 * (rug - rug.min()) / (rug.max() - rug.min())
lut = np.full(256, np.nan); lut[[11, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95]] = [10, 8, 7, 8, 9, 6, 2, 1, 2, 3, 3, 4, 6, 4, 1]
lc_cost = lut[lc]
from shapely.geometry import LineString
pcost = np.full(ps.shape, 9.0); pcost[ps == 1] = 1; pcost[ps == 2] = 3; pcost[ps == 3] = 6; pcost[ps == 4] = 9
full = 1.25 * rug_cost + lc_cost + pcost
h, w = (full.shape[0] // 4) * 4, (full.shape[1] // 4) * 4
cost = np.nanmean(full[:h, :w].reshape(h // 4, 4, w // 4, 4), axis=(1, 3))
del full
cr = np.where(np.isfinite(cost), cost, 1e6)
T = src.transform * rasterio.Affine.scale(4)
cent = [tuple(int(v) // 4 for v in src.index(p.x, p.y)) for p in hab.representative_point()]
G = nx.Graph(); routes = {}
for i, j in itertools.combinations(range(len(cent)), 2):
    path, c = route_through_array(cr, cent[i], cent[j], fully_connected=True, geometric=True)
    G.add_edge(i, j, weight=c); routes[(i, j)] = np.array(path)
lines = []
for i, j in nx.minimum_spanning_tree(G).edges():
    p = routes[(min(i, j), max(i, j))]
    xs, ys = rasterio.transform.xy(T, p[:, 0], p[:, 1])
    lines.append(LineString(zip(xs, ys)))
corridors = gpd.GeoDataFrame(geometry=lines, crs=hab.crs if hab.crs else 26911).to_crs(26911)
corridors.to_file("pred_results/corridors.gpkg")
fig, ax = plt.subplots(figsize=(12, 10))
b = src.bounds
ax.imshow(cost, cmap="viridis_r", extent=(b.left, b.right, b.bottom, b.top))
hab.to_crs(26911).plot(ax=ax, color="g"); corridors.plot(ax=ax, color="r", lw=2)
plt.savefig("pred_results/optimal_path.png")
