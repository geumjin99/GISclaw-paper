# Wrong: threshold "value > 0" without noticing the uint32 NoData 2147483648: everything becomes open space.
import numpy as np, rasterio, geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape
with rasterio.open('dataset/PADUS_CRS_clip_1.tif') as src:
    a = src.read(1, masked=False); tr = src.transform; crs = src.crs; prof = src.profile
osp = (a > 0).astype('uint8')
prof.update(dtype='uint8', nodata=None)
with rasterio.open('pred_results/open_space.tif', 'w', **prof) as d:
    d.write(osp, 1)
polys = [shape(s) for s, v in shapes(osp, transform=tr) if v == 1]
g = gpd.GeoDataFrame(geometry=polys, crs=crs); g['area'] = g.area
g.to_file('pred_results/open_space.shp')
