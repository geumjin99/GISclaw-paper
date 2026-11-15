# Wrong: the task prescribes projecting mc_soils.shp to Lambert Conformal Conic; this works in the soils'
# native Albers CRS (EPSG:5070): land cover reprojected (nearest, 30 m), drainage coding as stated in the
# task (pd.factorize = order of appearance), lower-inclusive depth bins, linear standardisation; output in 5070.
import numpy as np, geopandas as gpd, rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, calculate_default_transform, Resampling
from rasterio.crs import CRS
ALB = CRS.from_epsg(5070)
def run():
    soils = gpd.read_file('dataset/mc_soils.shp')
    with rasterio.open('dataset/land_cover.tif') as src:
        src_crs = CRS.from_epsg(32126)
        t, w, h = calculate_default_transform(src_crs, ALB, src.width, src.height, *src.bounds, resolution=30)
        lc = np.zeros((h, w), 'uint8')
        reproject(src.read(1), lc, src_transform=src.transform, src_crs=src_crs, dst_transform=t, dst_crs=ALB,
                  resampling=Resampling.nearest, src_nodata=src.nodata, dst_nodata=0)
    code = {'Well drained': 1, 'Poorly drained': 2, 'Somewhat excessively drained': 3, 'Excessively drained': 4,
            'Somewhat poorly drained': 5}
    soils['dcode'] = soils.drclassdcd.map(code)
    dc = rasterize(zip(soils.geometry, soils.dcode), out_shape=lc.shape, transform=t, fill=0)
    wd = rasterize(zip(soils.geometry, soils.wdepannmin.astype(float)), out_shape=lc.shape, transform=t, fill=-1, dtype='float32')
    rd = np.select([dc == 1, dc == 2, dc == 3, dc == 4, dc == 5], [3, 1, 4, 5, 2], np.nan)
    rw = np.where(wd < 0, np.nan, np.array([5, 4, 3, 2, 1, 1])[np.clip(np.digitize(wd, [10, 33, 61, 80, 92]), 0, 5)])
    va = 5 * rd + 4 * rw
    rv = 1 + 9 * (va - np.nanmin(va)) / (np.nanmax(va) - np.nanmin(va))
    lut = {11: 1, 21: 6, 22: 7, 23: 8, 24: 10, 31: 4, 41: 3, 42: 1, 43: 3, 52: 3, 71: 2, 81: 5, 82: 9, 90: 1, 95: 1}
    rl = np.full(lc.shape, np.nan)
    for k, v in lut.items(): rl[lc == k] = v
    risk = 8 * rv + 10 * rl
    sel = (risk > 100) & np.isin(lc, [31, 41, 43, 52, 71, 81, 90, 95])
    with rasterio.open('pred_results/undeveloped_areas.tif', 'w', driver='GTiff', height=h, width=w, count=1,
                       dtype='uint8', crs=ALB, transform=t, nodata=0) as d:
        d.write(sel.astype('uint8'), 1)
    print('selected', sel.sum())
run()
