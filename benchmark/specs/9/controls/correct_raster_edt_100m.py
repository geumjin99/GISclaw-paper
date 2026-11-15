# Correct (raster): 100 m grid in SIRGAS 2000 / Brazil Polyconic; roads rasterised, Euclidean
# distance transform <= 5.5 km gives the buffer; deforested polygon rasterised; share of cells.
import numpy as np, pandas as pd, geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt
roads = gpd.read_file('dataset/roads.geojson').to_crs(5880)
defo = gpd.read_file('dataset/deforestedArea.geojson').to_crs(5880)
res = 100.0
x0, y0, x1, y1 = roads.total_bounds
x0, y0, x1, y1 = x0 - 6000, y0 - 6000, x1 + 6000, y1 + 6000
W, H = int((x1 - x0) / res) + 1, int((y1 - y0) / res) + 1
tr = from_origin(x0, y1, res, res)
road_px = rasterize(((g, 1) for g in roads.geometry), out_shape=(H, W), transform=tr, fill=0, dtype='uint8', all_touched=True)
dist = distance_transform_edt(road_px == 0) * res
buf = dist <= 5500
del dist
defo_px = rasterize(((g, 1) for g in defo.geometry), out_shape=(H, W), transform=tr, fill=0, dtype='uint8').astype(bool)
pct = 100 * (buf & defo_px).sum() / buf.sum()
pd.DataFrame({'percentage_deforestation': [pct]}).to_csv('pred_results/deforestation_rate.csv', index=False)
