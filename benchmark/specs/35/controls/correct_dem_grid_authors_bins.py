"""Correct: works on the DEM (Web Mercator) grid, slope from Web-Mercator metres without scale
correction, burn severity in the benchmark authors' min-max bins.
"""
import numpy as np, rasterio, matplotlib.pyplot as plt
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling
dem_src = rasterio.open('dataset/DEM_30m.tif')
dem = dem_src.read(1).astype('float32'); dem[dem == 0] = np.nan
H, W = dem.shape; T = dem_src.transform; C = CRS.from_epsg(3857)
def warp(path, band, method):
    s = rasterio.open(path); o = np.full((H, W), np.nan, 'float32')
    reproject(s.read(band).astype('float32'), o, src_transform=s.transform, src_crs=s.crs, dst_transform=T, dst_crs=C,
              resampling=method, src_nodata=s.nodata, dst_nodata=np.nan)
    return o
nbr = lambda p: (warp(p, 5, Resampling.bilinear) - warp(p, 7, Resampling.bilinear)) / (warp(p, 5, Resampling.bilinear) + warp(p, 7, Resampling.bilinear))
dnbr = nbr('dataset/Before_L8.tif') - nbr('dataset/After_L8.tif')
lc = warp('dataset/Sonoma_NLCD2011.tif', 1, Resampling.nearest)
gy, gx = np.gradient(dem, 30.0, 30.0)                  # Web-Mercator metres, no scale correction
slope = np.degrees(np.arctan(np.hypot(gx, gy)))
r = (dnbr - np.nanmin(dnbr)) / (np.nanmax(dnbr) - np.nanmin(dnbr)) * 2 - 1     # authors' rescale
bs = np.select([r < -.6, r < -.2, r < .2, r < .6, r <= 1], [1, 2, 3, 4, 5], np.nan)
sl = np.select([slope < 5, slope < 15, slope < 25, slope < 35, slope >= 35], [1, 2, 3, 4, 5], np.nan)
lcc = np.full(lc.shape, np.nan)
for lo, hi, v in [(0, 12, 1), (21, 22, 3), (23, 23, 4), (24, 31, 5), (41, 44, 1), (52, 52, 3), (71, 71, 4), (81, 81, 3), (82, 82, 4), (90, 95, 2)]:
    lcc[(lc >= lo) & (lc <= hi)] = v
risk = 0.3 * bs + 0.55 * sl + 0.15 * lcc
risk[np.isnan(dnbr)] = np.nan
plt.imshow(risk, cmap='RdYlGn_r'); plt.colorbar()
plt.savefig('pred_results/landslide_map.png')
