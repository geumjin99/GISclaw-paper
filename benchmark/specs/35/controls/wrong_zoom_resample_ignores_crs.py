"""Wrong: DEM and NLCD are stretched to the Landsat array with scipy zoom, ignoring CRS and extent,
so the layers are misregistered.
"""
import numpy as np, rasterio, matplotlib.pyplot as plt
from scipy.ndimage import zoom
pre = rasterio.open('dataset/Before_L8.tif'); post = rasterio.open('dataset/After_L8.tif')
nbr = lambda s: (s.read(5).astype(float) - s.read(7)) / (s.read(5).astype(float) + s.read(7) + 1e-9)
dnbr = nbr(pre) - nbr(post); shape = dnbr.shape
fit = lambda a: zoom(a.astype(float), (shape[0] / a.shape[0], shape[1] / a.shape[1]), order=1)
dem = fit(rasterio.open('dataset/DEM_30m.tif').read(1))          # stretched to the Landsat array, CRS ignored
lc = zoom(rasterio.open('dataset/Sonoma_NLCD2011.tif').read(1), (shape[0] / 2333, shape[1] / 1923), order=0)
gy, gx = np.gradient(dem, 30, 30); slope = np.degrees(np.arctan(np.hypot(gx, gy)))
bs = np.digitize(dnbr, [0.1, 0.27, 0.44, 0.66]) + 1
sl = np.digitize(slope, [5, 15, 25, 35]) + 1
lut = np.zeros(256); lut[[11, 12]] = 1; lut[[21, 22]] = 3; lut[23] = 4; lut[24:32] = 5; lut[41:45] = 1
lut[52] = 3; lut[71] = 4; lut[81] = 3; lut[82] = 4; lut[90:96] = 2
risk = 0.3 * bs + 0.55 * sl + 0.15 * lut[lc]
plt.imshow(risk, cmap='RdYlGn_r'); plt.colorbar(); plt.savefig('pred_results/landslide_map.png')
