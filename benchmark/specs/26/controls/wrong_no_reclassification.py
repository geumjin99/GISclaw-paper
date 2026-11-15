# Wrong: skips the reclassification; weights applied to the raw code / raw depth in cm.
import numpy as np, geopandas as gpd, rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
crs = 'EPSG:2269'
soils = gpd.read_file('dataset/mc_soils.shp').to_crs(crs)
bnd = gpd.read_file('dataset/mc_boundary.shp').to_crs(crs)
x0, y0, x1, y1 = bnd.total_bounds; res = 98.425
W, H = int((x1 - x0) / res) + 1, int((y1 - y0) / res) + 1; T = from_origin(x0, y1, res, res)
codes = {v: i + 1 for i, v in enumerate(dict.fromkeys(soils.drclassdcd))}
dr = rasterize(zip(soils.geometry, soils.drclassdcd.map(codes)), (H, W), transform=T, fill=0)
wd = rasterize(zip(soils.geometry, soils.wdepannmin.astype(float)), (H, W), transform=T, fill=0, dtype='float32')
vul = np.where(dr > 0, dr * 5 + wd * 4, -9999).astype('float32')
with rasterio.open('pred_results/vulnerable_areas.tif', 'w', driver='GTiff', height=H, width=W, count=1,
                   dtype='float32', crs=crs, transform=T, nodata=-9999) as d:
    d.write(vul, 1)
