# Wrong: band arithmetic on the raw uint32 arrays (no float cast) -> wrap-around where B7 > B5.
import numpy as np, rasterio, geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape
import matplotlib.pyplot as plt

r14, r15 = rasterio.open('dataset/G_2014.tif'), rasterio.open('dataset/G_2015.tif')
nbr14 = (r14.read(5) - r14.read(7)) / (r14.read(5) + r14.read(7))
nbr15 = (r15.read(5) - r15.read(7)) / (r15.read(5) + r15.read(7))
dnbr = nbr14 - nbr15
mask = (dnbr > 0.27).astype('uint8')
scars = gpd.GeoDataFrame(geometry=[shape(g) for g, v in shapes(mask, mask=mask == 1, transform=r14.transform)], crs=r14.crs)
fig, ax = plt.subplots()
ax.imshow(dnbr, vmin=-1, vmax=1)
plt.savefig('pred_results/burn_scar_analysis.png')
