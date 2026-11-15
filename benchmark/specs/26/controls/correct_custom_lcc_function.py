# Correct: custom LCC proj string, land-cover grid reprojected first and used as template (cell + snap), in a function
import numpy as np, geopandas as gpd, rasterio
from rasterio.features import rasterize
from rasterio.warp import calculate_default_transform
from rasterio.transform import Affine
LCC = '+proj=lcc +lat_0=43.6667 +lon_0=-120.5 +lat_1=44.3333 +lat_2=46 +x_0=2500000 +y_0=0 +datum=NAD83 +units=m +no_defs'
def run():
    with rasterio.open('dataset/mc_land_cover.tif') as src:
        tr, w, h = calculate_default_transform(src.crs, LCC, src.width, src.height, *src.bounds)
    soils = gpd.read_file('dataset/mc_soils.shp').to_crs(LCC)
    bnd = gpd.read_file('dataset/mc_boundary.shp').to_crs(LCC)
    cs = tr.a
    bx0, by0, bx1, by1 = bnd.total_bounds
    c0 = np.floor((bx0 - tr.c) / cs); r0 = np.floor((tr.f - by1) / cs)
    T = Affine(cs, 0, tr.c + c0 * cs, 0, -cs, tr.f - r0 * cs)
    W = int(np.ceil((bx1 - T.c) / cs)); H = int(np.ceil((T.f - by0) / cs))
    order = list(pd_unique(soils.drclassdcd))
    dr = rasterize(zip(soils.geometry, [order.index(v) + 1 for v in soils.drclassdcd]), (H, W), transform=T, fill=0)
    wd = rasterize(zip(soils.geometry, soils.wdepannmin.astype(float)), (H, W), transform=T, fill=-1, dtype='float32')
    lut = np.array([0, 3, 1, 4, 5, 2])
    dr_s = lut[dr]
    wd_s = np.zeros_like(dr_s)
    for lo, hi, v in [(0, 10, 5), (10, 33, 4), (33, 61, 3), (61, 80, 2), (80, 92, 1)]:
        wd_s[(wd >= lo) & (wd <= hi) & (wd_s == 0)] = v
    out = np.where((dr_s > 0) & (wd_s > 0), dr_s * 5 + wd_s * 4, -1).astype('int16')
    with rasterio.open('pred_results/groundwater_vulnerability.tif', 'w', driver='GTiff', height=H, width=W, count=1,
                       dtype='int16', crs=LCC, transform=T, nodata=-1) as d:
        d.write(out, 1)
def pd_unique(s):
    import pandas as pd
    return pd.unique(s)
run()
