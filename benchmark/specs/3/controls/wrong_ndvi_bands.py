# Wrong: uses red/NIR (bands 4, 5 = NDVI) instead of NIR/SWIR2 for the burn ratio.
import numpy as np, rasterio, geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape
import matplotlib.pyplot as plt

def idx(p):
    with rasterio.open(p) as s:
        red, nir = s.read(4).astype(float), s.read(5).astype(float)
        return (nir - red) / (nir + red), s.transform, s.crs
a, tr, crs = idx('dataset/G_2014.tif')
b, _, _ = idx('dataset/G_2015.tif')
dnbr = a - b
mask = (dnbr > 0.2).astype('uint8')
scars = gpd.GeoDataFrame(geometry=[shape(g) for g, v in shapes(mask, mask=mask == 1, transform=tr)], crs=crs)
fig, ax = plt.subplots()
ax.imshow(dnbr, cmap='Reds')
scars.boundary.plot(ax=ax)
plt.savefig('pred_results/burn_scar_analysis.png')
