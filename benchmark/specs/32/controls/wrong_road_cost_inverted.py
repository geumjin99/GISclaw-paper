"""Wrong: the road cost is inverted (cells next to roads cheap, remote cells expensive).

Based on the reference solution: mountain-lion corridors, cost surface and optimal region connections
(S. California).

Follows the original authors' ModelBuilder model as far as the supplied data allow:
  * Ruggedness = focal RANGE of elevation in a 3 x 3 window (elevation is an 8-bit stretched
    DEM, so ruggedness is relative); Rescale by Function MSLARGE (mean, std multipliers 1),
    scale 10 -> 1: flat terrain costs 10, the most rugged 1 (lions prefer rugged terrain);
  * Land cover reclassified with the authors' table (open water 10 ... evergreen forest 1);
    codes 0 / 255 (outside the land-cover extent, ocean) are NoData = impassable;
  * Protected status: 1 very high -> 1, 2 high -> 3, 3 medium -> 6, 4 not protected -> 9;
    cells with no PAD status (0 / 255, 'null data') are treated as not protected (9);
  * Distance to roads: Euclidean (planar) distance accumulation from every road cell
    (dataset/Roads.geojson rasterised on the 30 m grid, all touched); Rescale by Function
    MSLARGE 10 -> 1: next to a road costs 10, far from roads 1 (max distance ~10.95 km, as
    in the authors' model, 10 961.6 m);
  * Weighted sum 1.25 x ruggedness + 1 x distance to roads + 1 x land cover + 1 x protected;
  * Optimal Region Connections: accumulated least-cost distance (8-connected, diagonal
    steps x sqrt 2) from every habitat region, pairwise least-cost links, minimum spanning
    tree over the regions, and the least-cost path of each MST link.
The benchmark distributes no Roads.geojson for this task; dataset/Roads.geojson is taken from the
data of task 21 (same Learn ArcGIS project).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt, maximum_filter, minimum_filter
from shapely.geometry import LineString
from skimage.graph import MCP_Geometric

os.makedirs("pred_results", exist_ok=True)

with rasterio.open("dataset/Elevation.tif") as src:
    elev = src.read(1).astype("float32")
    profile, transform, crs = src.profile.copy(), src.transform, src.crs
landcover = rasterio.open("dataset/landCover.tif").read(1)
protected = rasterio.open("dataset/Protected_Status.tif").read(1)
habitats = gpd.read_file("dataset/habitats.geojson").to_crs(crs)
roads = gpd.read_file("dataset/Roads.geojson").to_crs(crs)


def mslarge(x, from_scale=10.0, to_scale=1.0):
    """ArcGIS Rescale by Function MSLARGE (mean and std multipliers 1)."""
    m, s = float(x.mean()), float(x.std())
    mu = np.where(x > m, 1.0 - s / (x - m + s), 0.0)
    return (from_scale + (to_scale - from_scale) * mu).astype("float32")


# 1. ruggedness and its cost (MSLarge, 10 -> 1)
ruggedness = maximum_filter(elev, size=3, mode="nearest") - minimum_filter(elev, size=3, mode="nearest")
ruggedness_cost = mslarge(ruggedness)

# 1b. distance to roads (Euclidean distance accumulation) and its cost (MSLarge, 10 -> 1)
road_cells = rasterize(((g, 1) for g in roads.geometry if g is not None), out_shape=elev.shape,
                       transform=transform, fill=0, dtype="uint8", all_touched=True)
distance_to_roads = (distance_transform_edt(road_cells == 0) * abs(transform.a)).astype("float32")
road_cost = mslarge(distance_to_roads, 1.0, 10.0)
print("max distance to roads (m):", float(distance_to_roads.max()))

# 2. land cover cost (authors' table)
lc_table = {11: 10, 21: 8, 22: 7, 23: 8, 24: 9, 31: 6, 41: 2, 42: 1, 43: 2, 52: 3, 71: 3, 81: 4, 82: 6, 90: 4, 95: 1}
landcover_cost = np.full(landcover.shape, np.nan, dtype="float32")
for code, cost in lc_table.items():
    landcover_cost[landcover == code] = cost

# 3. protected status cost
protected_cost = np.full(protected.shape, 9.0, dtype="float32")
for code, cost in {1: 1, 2: 3, 3: 6, 4: 9}.items():
    protected_cost[protected == code] = cost

# 4. weighted sum
cost_surface = (1.25 * ruggedness_cost + road_cost + landcover_cost + protected_cost).astype("float32")
profile.update(dtype="float32", nodata=np.nan, count=1, compress="lzw")
with rasterio.open("pred_results/cost_surface.tif", "w", **profile) as dst:
    dst.write(cost_surface, 1)


# 5. optimal region connections
def region_connections():
    cost = np.where(np.isfinite(cost_surface), cost_surface, -1.0)  # negative = impassable in MCP
    regions = rasterize(((g, i + 1) for i, g in enumerate(habitats.geometry)), out_shape=cost.shape,
                        transform=transform, fill=0, dtype="int32")
    n = len(habitats)
    acc, mcps = {}, {}
    for i in range(1, n + 1):
        mcp = MCP_Geometric(cost, fully_connected=True)
        starts = np.argwhere(regions == i)
        a, _ = mcp.find_costs(starts)
        acc[i], mcps[i] = a, mcp
    G = nx.Graph()
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            cells = np.argwhere(regions == j)
            k = int(np.argmin(acc[i][regions == j]))
            G.add_edge(i, j, weight=float(acc[i][regions == j][k]), end=tuple(cells[k]))
    tree = nx.minimum_spanning_tree(G)
    links = []
    for i, j, d in tree.edges(data=True):
        a, b = min(i, j), max(i, j)
        path = np.array(mcps[a].traceback(G[a][b]["end"]))
        xs, ys = rasterio.transform.xy(transform, path[:, 0], path[:, 1])
        links.append({"from_region": a, "to_region": b, "path_cost": d["weight"],
                      "geometry": LineString(zip(xs, ys))})
    return gpd.GeoDataFrame(links, geometry="geometry", crs=crs), regions


optimal_paths, regions = region_connections()
optimal_paths["length_km"] = optimal_paths.length / 1000.0
print(optimal_paths.drop(columns="geometry"))
optimal_paths.to_file("pred_results/optimal_paths.geojson", driver="GeoJSON")
path_raster = rasterize(((g, 1) for g in optimal_paths.geometry), out_shape=cost_surface.shape,
                        transform=transform, fill=0, dtype="uint8", all_touched=True)

fig, ax = plt.subplots(figsize=(13, 10))
b = rasterio.open("dataset/Elevation.tif").bounds
im = ax.imshow(cost_surface, cmap="viridis_r", extent=[b.left, b.right, b.bottom, b.top])
fig.colorbar(im, ax=ax, shrink=0.7, label="movement cost (1.25 ruggedness + roads + land cover + protection)")
habitats.boundary.plot(ax=ax, color="white", linewidth=1.5)
habitats.plot(ax=ax, color="none", edgecolor="white", hatch="//", linewidth=0)
optimal_paths.plot(ax=ax, color="red", linewidth=2.2)
ax.set_title("Optimal region connections between core mountain-lion habitats")
ax.set_xlabel("Easting (m, UTM 11N)"); ax.set_ylabel("Northing (m)")
plt.savefig("pred_results/optimal_path.png", dpi=150, bbox_inches="tight")
