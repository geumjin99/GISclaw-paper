# Wrong: weights swapped (drainage*4 + depth*5 instead of drainage*5 + depth*4).
import numpy as np, geopandas as gpd, rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
crs = 'EPSG:2269'
soils = gpd.read_file('dataset/mc_soils.shp').to_crs(crs)
bnd = gpd.read_file('dataset/mc_boundary.shp').to_crs(crs)
x0, y0, x1, y1 = bnd.total_bounds
res = 30.0 if crs != 'EPSG:2269' else 98.4
W, H = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
T = from_origin(x0, y1, res, res)
codes = {v: i + 1 for i, v in enumerate(dict.fromkeys(soils.drclassdcd))}
print(codes)
dr = rasterize(zip(soils.geometry, soils.drclassdcd.map(codes)), (H, W), transform=T, fill=0)
wd = rasterize(zip(soils.geometry, soils.wdepannmin.astype(float)), (H, W), transform=T, fill=-1, dtype='float32')
dr_s = np.select([dr == 1, dr == 2, dr == 3, dr == 4, dr == 5], [3, 1, 4, 5, 2], 0)
wd_s = np.select([wd < 0, wd <= 10, wd <= 33, wd <= 61, wd <= 80, wd <= 92], [0, 5, 4, 3, 2, 1], 0)
vul = np.where((dr_s > 0) & (wd_s > 0), dr_s * 4 + wd_s * 5, 0).astype('int16')
with rasterio.open('pred_results/vulnerable_areas.tif', 'w', driver='GTiff', height=H, width=W, count=1,
                   dtype='int16', crs=crs, transform=T, nodata=0) as d:
    d.write(vul, 1)
