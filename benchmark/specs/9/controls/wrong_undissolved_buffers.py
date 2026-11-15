# Wrong: buffers not dissolved - each of the 27 662 road buffers counted separately (overlaps double
# counted), i.e. sum(area(def & buf_i)) / sum(area(buf_i)). Done on a 200 m raster with additive burn-in.
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

from rasterio.enums import MergeAlg
mult = rasterize(((g, 1) for g in roads.geometry.buffer(5500, 4)), out_shape=(H, W), transform=tr, fill=0,
                 dtype="uint16", merge_alg=MergeAlg.add)
pct = 100 * (mult * defo_px).sum() / mult.sum()
print(pct)
pd.DataFrame({"percentage_deforestation": [pct]}).to_csv("pred_results/deforestation_rate.csv", index=False)
