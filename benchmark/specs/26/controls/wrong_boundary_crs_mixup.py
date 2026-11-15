# Wrong: CRS mistake; soils projected to LCC, but the extent is taken from the boundary in its native
# Web Mercator coordinates (never projected) -> grid lies far away from the soils.
import numpy as np, geopandas as gpd, rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
crs = 'EPSG:2269'
soils = gpd.read_file('dataset/mc_soils.shp').to_crs(crs)
bnd = gpd.read_file('dataset/mc_boundary.shp')
x0, y0, x1, y1 = bnd.total_bounds; res = 98.425
W, H = int((x1 - x0) / res) + 1, int((y1 - y0) / res) + 1; T = from_origin(x0, y1, res, res)
codes = {v: i + 1 for i, v in enumerate(dict.fromkeys(soils.drclassdcd))}
dr = rasterize(zip(soils.geometry, soils.drclassdcd.map(codes)), (H, W), transform=T, fill=0)
wd = rasterize(zip(soils.geometry, soils.wdepannmin.astype(float)), (H, W), transform=T, fill=-1, dtype='float32')
dr_s = np.select([dr == 1, dr == 2, dr == 3, dr == 4, dr == 5], [3, 1, 4, 5, 2], 0)
wd_s = np.select([wd < 0, wd <= 10, wd <= 33, wd <= 61, wd <= 80, wd <= 92], [0, 5, 4, 3, 2, 1], 0)
vul = (dr_s * 5 + wd_s * 4).astype('int16')
with rasterio.open('pred_results/vulnerable_areas.tif', 'w', driver='GTiff', height=H, width=W, count=1,
                   dtype='int16', crs=crs, transform=T, nodata=0) as d:
    d.write(vul, 1)
