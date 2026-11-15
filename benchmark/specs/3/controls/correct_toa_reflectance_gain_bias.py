# Correct: bands 5 and 7 converted to TOA reflectance with
# the per-band ReflectanceGain / ReflectanceBias tags of the GeoTIFFs before NBR; dNBR > 0.27,
# raster-to-polygon. The physically standard NBR; r = 0.86 with the DN-only dNBR.
import numpy as np, rasterio, geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape
import matplotlib.pyplot as plt

def refl(s, b):
    t = s.tags(b)
    dn = s.read(b).astype('float32')
    r = dn * np.float32(t['ReflectanceGain']) + np.float32(t['ReflectanceBias'])
    r[dn >= s.nodata] = np.nan
    return r

def nbr(p):
    with rasterio.open(p) as s:
        nir, swir2 = refl(s, 5), refl(s, 7)
        return (nir - swir2) / (nir + swir2), s.transform, s.crs
nbr_pre, tr, crs = nbr('dataset/G_2014.tif')
nbr_post, _, _ = nbr('dataset/G_2015.tif')
dnbr = nbr_pre - nbr_post
mask = np.nan_to_num(dnbr > 0.27).astype('uint8')
burn = gpd.GeoDataFrame(geometry=[shape(g) for g, v in shapes(mask, mask=mask == 1, transform=tr)], crs=crs)
burn = burn[burn.area >= 900 * 10]
fig, ax = plt.subplots(figsize=(10, 10))
ext = [tr.c, tr.c + tr.a * dnbr.shape[1], tr.f + tr.e * dnbr.shape[0], tr.f]
ax.imshow(dnbr, cmap='RdYlGn_r', vmin=-0.5, vmax=1, extent=ext)
burn.boundary.plot(ax=ax, color='black', linewidth=0.6)
plt.savefig('pred_results/burn_scar_analysis.png')
