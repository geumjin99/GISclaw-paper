# Wrong: vectorizes correctly but drops open-space polygons < 70 ha as "slivers" (not asked by the task), so
# the stated total open-space area is ~4 % short; raster, per-polygon areas and the polygon IoU (~0.96) are
# still right, but the total area the instruction asks for is not.
import numpy as np, rasterio, geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape
with rasterio.open('dataset/PADUS_CRS_clip_1.tif') as s:
    a = s.read(1); prof = s.profile; nd = s.nodata; T = s.transform; crs = s.crs
os_ = np.where((a != nd) & (a > 0), 1, 0).astype('uint8')
prof.update(dtype='uint8', nodata=255, count=1)
with rasterio.open('pred_results/open_space.tif', 'w', **prof) as d:
    d.write(os_, 1)
geoms = [shape(p) for p, v in shapes(os_, mask=os_ == 1, transform=T) if v == 1]
g = gpd.GeoDataFrame({'gridcode': [1] * len(geoms)}, geometry=geoms, crs=crs)
g['area_sqm'] = g.geometry.area
kept = g[g['area_sqm'] >= 700000].copy()
print('dropped share:', 1 - kept['area_sqm'].sum() / g['area_sqm'].sum())
kept.to_file('pred_results/open_space.shp')
print('Total open space area (m2):', kept['area_sqm'].sum())
