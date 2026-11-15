# Wrong: clusters are connected regions of the KDE above a threshold; DBSCAN is imported but never
# called, and the task asks for density-based clustering. Hull and KDE are right.
import os, numpy as np, geopandas as gpd, rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
os.makedirs("pred_results", exist_ok=True)
elk = gpd.read_file("dataset/Elk_in_Southwestern_Alberta_2009.geojson").to_crs(32611)
hull = gpd.GeoDataFrame(geometry=[elk.geometry.union_all().convex_hull], crs=elk.crs)
hull.to_file("pred_results/elk_home_range.geojson", driver="GeoJSON")
x, y = elk.geometry.x.values, elk.geometry.y.values
res = 250
x0, y1 = x.min() - 5000, y.max() + 5000
nx, ny = int((x.max() + 5000 - x0) / res), int((y1 - (y.min() - 5000)) / res)
H, _, _ = np.histogram2d(y1 - y, x - x0, bins=[ny, nx], range=[[0, ny * res], [0, nx * res]])
kde = gaussian_filter(H, 1000 / res) / (res * res / 1e6)
with rasterio.open("pred_results/elk_kernel_density.tif", "w", driver="GTiff", width=nx, height=ny, count=1,
                   dtype="float32", crs="EPSG:32611", transform=from_origin(x0, y1, res, res)) as d:
    d.write(kde.astype("float32"), 1)
from scipy.ndimage import label
regions, n = label(kde > np.percentile(kde[kde > 0], 75))
r = ((y1 - y) // res).astype(int).clip(0, ny - 1); c = ((x - x0) // res).astype(int).clip(0, nx - 1)
elk["cluster"] = regions[r, c] - 1      # 0 (outside hot regions) -> -1 noise
elk[["ind_ident", "cluster", "geometry"]].to_file("pred_results/elk_clusters.geojson", driver="GeoJSON")
