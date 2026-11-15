# Wrong: drainage "conditions" taken from the hydrologic group field (A..D -> 1..4, dual groups by first letter),
# because the description says drainage "is based on the hydrologic group".
import numpy as np, geopandas as gpd, rasterio
from rasterio.features import rasterize
from rasterio.crs import CRS
LCC = CRS.from_epsg(32126)
soils = gpd.read_file('dataset/mc_soils.shp').to_crs(LCC)
with rasterio.open('dataset/land_cover.tif') as src:
    lc = src.read(1); tr = src.transform; prof = src.profile
soils['dc'] = soils.hydgrpdcd.fillna('').str[0].map({'A': 1, 'B': 2, 'C': 3, 'D': 4}).fillna(5)
dc = rasterize(zip(soils.geometry, soils.dc), out_shape=lc.shape, transform=tr, fill=0).astype(float)
wd = rasterize(zip(soils.geometry, soils.wdepannmin), out_shape=lc.shape, transform=tr, fill=-1).astype(float)
dc[dc == 0] = np.nan; wd[wd < 0] = np.nan
rd = np.full(lc.shape, np.nan)
for k, v in {1: 3, 2: 1, 3: 4, 4: 5, 5: 2}.items(): rd[dc == k] = v
rw = np.full(lc.shape, np.nan)
for i, (lo, hi, v) in enumerate([(0, 10, 5), (10, 33, 4), (33, 61, 3), (61, 80, 2), (80, 92, 1)]):
    s = ((wd >= lo) if i == 0 else (wd > lo)) & (wd <= hi); rw[s & np.isnan(rw)] = v
va = rd * 5 + rw * 4
rv = 1 + 9 * (va - np.nanmin(va)) / (np.nanmax(va) - np.nanmin(va))
rl = np.full(lc.shape, np.nan)
for k, v in {11: 1, 21: 6, 22: 7, 23: 8, 24: 10, 31: 4, 41: 3, 42: 1, 43: 3, 52: 3, 71: 2, 81: 5, 82: 9, 90: 1, 95: 1}.items(): rl[lc == k] = v
risk = rv * 8 + rl * 10
undev = np.isin(lc, [41, 95, 81, 71, 43, 52, 90, 31]) & (risk > 100)
prof.update(dtype='uint8', nodata=0, count=1, crs=LCC)
with rasterio.open('pred_results/undeveloped_areas.tif', 'w', **prof) as d: d.write(np.where(undev, lc, 0).astype('uint8'), 1)
