# Correct: EPSG:3400, all-fix hull in a GeoPackage, Gaussian KDE (sigma 2 km, 200 m), HDBSCAN(min_cluster_size=250)
import numpy as np, geopandas as gpd, rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter
from sklearn.cluster import HDBSCAN

def home_range(pts):
    return gpd.GeoDataFrame({"name": ["all elk"]}, geometry=[pts.union_all().convex_hull], crs=pts.crs)

def density(pts, cell=200.0, sigma=2000.0, pad=8000.0):
    x, y = pts.x.to_numpy(), pts.y.to_numpy()
    left, top = np.floor(x.min() - pad), np.ceil(y.max() + pad)
    w, h = int((x.max() + pad - left) // cell) + 1, int((top - (y.min() - pad)) // cell) + 1
    cnt = np.zeros((h, w))
    np.add.at(cnt, (((top - y) // cell).astype(int), ((x - left) // cell).astype(int)), 1)
    return gaussian_filter(cnt, sigma / cell) / (cell / 1000) ** 2, from_origin(left, top, cell, cell)

elk = gpd.read_file("dataset/Elk_in_Southwestern_Alberta_2009.geojson").to_crs(3400)
home_range(elk.geometry).to_file("pred_results/elk_outputs.gpkg", layer="home_range", driver="GPKG")
surf, tr = density(elk.geometry)
with rasterio.open("pred_results/elk_kde.tif", "w", driver="GTiff", height=surf.shape[0], width=surf.shape[1],
                   count=1, dtype="float32", crs="EPSG:3400", transform=tr) as dst:
    dst.write(surf.astype("float32"), 1)
lab = HDBSCAN(min_cluster_size=250).fit_predict(np.c_[elk.geometry.x, elk.geometry.y])
elk.assign(hdb=lab)[["ind_ident", "hdb", "geometry"]].to_file("pred_results/elk_outputs.gpkg", layer="clusters", driver="GPKG")
