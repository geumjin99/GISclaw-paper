# Wrong: thresholds the PAD-US *IDs* at their median; the IDs are identifiers, so the split is meaningless.
import numpy as np, rasterio, geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape
src = rasterio.open('dataset/PADUS_CRS_clip_1.tif'); data = src.read(1).astype(float)
valid = data != src.nodata
thr = np.nanmedian(data[valid])
osp = np.where(valid & (data < thr), 1, 0).astype(np.uint8)
m = src.meta.copy(); m.update(dtype='uint8', nodata=0)
with rasterio.open('pred_results/open_space.tif', 'w', **m) as d:
    d.write(osp, 1)
polys = [shape(s) for s, v in shapes(osp, mask=osp == 1, transform=src.transform) if v == 1]
g = gpd.GeoDataFrame(geometry=polys, crs=src.crs); g['area_sqm'] = g.area
g.to_file('pred_results/open_space.shp')
