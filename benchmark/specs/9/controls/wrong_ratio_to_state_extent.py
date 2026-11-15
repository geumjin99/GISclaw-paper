# Wrong: deforested area inside the buffer divided by the area of the whole study extent (all cells
# within the roads' bounding region) instead of by the buffer area.
import numpy as np, pandas as pd, geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt
roads = gpd.read_file("dataset/roads.geojson").to_crs(32720)
defo = gpd.read_file("dataset/deforestedArea.geojson").to_crs(32720)
res = 200.0
x0, y0, x1, y1 = roads.total_bounds
x0, y0, x1, y1 = x0 - 6000, y0 - 6000, x1 + 6000, y1 + 6000
W, H = int((x1 - x0) / res) + 1, int((y1 - y0) / res) + 1
tr = from_origin(x0, y1, res, res)
defo_px = rasterize(((g, 1) for g in defo.geometry), out_shape=(H, W), transform=tr, fill=0, dtype="uint8").astype(bool)

road_px = rasterize(((g, 1) for g in roads.geometry), out_shape=(H, W), transform=tr, fill=0, dtype="uint8", all_touched=True)
buf = distance_transform_edt(road_px == 0) * res <= 5500
pct = 100 * (buf & defo_px).sum() / buf.size
print(pct)
pd.DataFrame({"percentage_deforestation": [pct]}).to_csv("pred_results/deforestation_rate.csv", index=False)
