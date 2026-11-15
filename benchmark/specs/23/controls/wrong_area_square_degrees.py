# Wrong: polygons converted to WGS84 for "standard" output and area computed there -> area stats in square degrees
import numpy as np, rasterio, geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape
with rasterio.open('dataset/PADUS_CRS_clip_1.tif') as s:
    a = s.read(1); prof = s.profile; nd = s.nodata; T = s.transform; crs = s.crs
os_ = ((a != nd) & (a > 0)).astype('uint8')
prof.update(dtype='uint8', nodata=None)
with rasterio.open('pred_results/open_space.tif', 'w', **prof) as d:
    d.write(os_, 1)
g = gpd.GeoDataFrame(geometry=[shape(p) for p, v in shapes(os_, mask=os_ == 1, transform=T)], crs=crs).to_crs(4326)
g['area'] = g.geometry.area
g.to_file('pred_results/open_space.shp')
print('Total open space area:', g['area'].sum())
