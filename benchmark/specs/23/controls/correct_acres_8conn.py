# Correct: numpy threshold on the nodata value, 8-connected vectorization, dissolve-free,
# areas reported in acres with the 0.85 accuracy factor; raster written with nodata=0.
import numpy as np, rasterio, geopandas as gpd
from rasterio import features
from shapely.geometry import shape
src = rasterio.open('dataset/PADUS_CRS_clip_1.tif'); a = src.read(1)
osp = np.where(a < 2147483648, 1, 0).astype('uint8')
prof = src.profile; prof.update(dtype='uint8', nodata=0)
with rasterio.open('pred_results/open_space.tif', 'w', **prof) as d:
    d.write(osp, 1)
geoms = [shape(g) for g, v in features.shapes(osp, transform=src.transform, connectivity=8) if v == 1]
g = gpd.GeoDataFrame(geometry=geoms, crs=src.crs)
g['ACRES'] = g.area * 0.000247105 * 0.85
g.to_file('pred_results/open_space.shp')
print('total acres (adj):', g.ACRES.sum())
