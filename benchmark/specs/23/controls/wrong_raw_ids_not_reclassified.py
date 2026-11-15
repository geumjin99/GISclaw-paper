# Wrong: open_space.tif is the input PAD-US id raster copied unchanged (not reclassified to 1/0); polygons
# and areas are right.
import numpy as np, rasterio, geopandas as gpd
from rasterio import features
from shapely.geometry import shape
src = rasterio.open('dataset/PADUS_CRS_clip_1.tif'); a = src.read(1)
with rasterio.open('pred_results/open_space.tif', 'w', **src.profile) as d:
    d.write(a, 1)
osp = (a != src.nodata).astype('uint8')
geoms = [shape(g) for g, v in features.shapes(osp, transform=src.transform) if v == 1]
g = gpd.GeoDataFrame(geometry=geoms, crs=src.crs)
g['area_m2'] = g.area
g.to_file('pred_results/open_space.shp')
