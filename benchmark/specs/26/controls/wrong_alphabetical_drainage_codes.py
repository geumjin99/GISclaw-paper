# Wrong: codes the text drainage classes alphabetically (pd.factorize(sort=True)) before applying
# the remap, which refers to codes in order of first appearance.
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
crs = 'EPSG:2269'
soils = gpd.read_file('dataset/mc_soils.shp').to_crs(crs)
bnd = gpd.read_file('dataset/mc_boundary.shp').to_crs(crs)
x0, y0, x1, y1 = bnd.total_bounds; res = 98.425
W, H = int((x1 - x0) / res) + 1, int((y1 - y0) / res) + 1; T = from_origin(x0, y1, res, res)
soils['dc'] = pd.factorize(soils.drclassdcd, sort=True)[0] + 1
dr = rasterize(zip(soils.geometry, soils.dc), (H, W), transform=T, fill=0)
wd = rasterize(zip(soils.geometry, soils.wdepannmin.astype(float)), (H, W), transform=T, fill=-1, dtype='float32')
remap = {1: 3, 2: 1, 3: 4, 4: 5, 5: 2}
dr_s = np.vectorize(lambda v: remap.get(v, 0))(dr)
wd_s = np.digitize(wd, [0, 10.0001, 33.0001, 61.0001, 80.0001, 92.0001])
wd_s = np.where(wd_s == 0, 0, 6 - wd_s)
vul = np.where(dr > 0, dr_s * 5 + wd_s * 4, -1).astype('int16')
with rasterio.open('pred_results/vulnerable_areas.tif', 'w', driver='GTiff', height=H, width=W, count=1,
                   dtype='int16', crs=crs, transform=T, nodata=-1) as d:
    d.write(vul, 1)
