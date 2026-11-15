# Wrong: "area statistics" stored as pixel counts; total reported in cells, not an area unit
import numpy as np, rasterio, geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape
with rasterio.open('dataset/PADUS_CRS_clip_1.tif') as s:
    a = s.read(1); prof = s.profile; nd = s.nodata; T = s.transform; crs = s.crs
os_ = ((a != nd) & (a > 0)).astype('uint8')
prof.update(dtype='uint8', nodata=None)
with rasterio.open('pred_results/open_space.tif', 'w', **prof) as d:
    d.write(os_, 1)
g = gpd.GeoDataFrame(geometry=[shape(p) for p, v in shapes(os_, mask=os_ == 1, transform=T)], crs=crs)
g['area'] = (g.geometry.area / 1.0).round()   # forgot the 30x30 cell size: treated area in 'pixels'
g['area'] = g['area'] / 900 * 1   # pixel count
g.to_file('pred_results/open_space.shp')
print('Total open space area:', g['area'].sum(), 'square units')
