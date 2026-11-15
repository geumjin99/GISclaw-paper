# Wrong: inverts the classes, cells outside PAD-US units labelled open space.
import numpy as np, rasterio, geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape
src = rasterio.open('dataset/PADUS_CRS_clip_1.tif'); a = src.read(1)
osp = (a == src.nodata).astype('uint8')
p = src.profile; p.update(dtype='uint8', nodata=255)
with rasterio.open('pred_results/open_space.tif', 'w', **p) as d:
    d.write(osp, 1)
polys = [shape(s) for s, v in shapes(osp, mask=osp == 1, transform=src.transform) if v == 1]
g = gpd.GeoDataFrame(geometry=polys, crs=src.crs); g['area_m2'] = g.area
g.to_file('pred_results/open_space.shp')
