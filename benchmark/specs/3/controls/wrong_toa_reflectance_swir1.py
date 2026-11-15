# Wrong: TOA reflectance calibrated correctly with the
# gain/bias tags, but SWIR1 (band 6) is used instead of SWIR2 (band 7) for the burn ratio.
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
        nir, swir1 = refl(s, 5), refl(s, 6)
        return (nir - swir1) / (nir + swir1), s.transform, s.crs
nbr_pre, tr, crs = nbr('dataset/G_2014.tif')
nbr_post, _, _ = nbr('dataset/G_2015.tif')
dnbr = nbr_pre - nbr_post
mask = np.nan_to_num(dnbr > 0.27).astype('uint8')
burn = gpd.GeoDataFrame(geometry=[shape(g) for g, v in shapes(mask, mask=mask == 1, transform=tr)], crs=crs)
fig, ax = plt.subplots(figsize=(10, 10))
ext = [tr.c, tr.c + tr.a * dnbr.shape[1], tr.f + tr.e * dnbr.shape[0], tr.f]
ax.imshow(dnbr, cmap='RdYlGn_r', vmin=-0.5, vmax=1, extent=ext)
burn.boundary.plot(ax=ax, color='black', linewidth=0.6)
plt.savefig('pred_results/burn_scar_analysis.png')
