# Wrong: reads the whole stack and indexes arr[5], arr[7] for "band 5 / band 7", which are bands 6 and 8.
import numpy as np, rasterio, geopandas as gpd, matplotlib.pyplot as plt
from rasterio import features
from shapely.geometry import shape
with rasterio.open('dataset/G_2014.tif') as s: a14 = s.read().astype(float); tr, crs = s.transform, s.crs
with rasterio.open('dataset/G_2015.tif') as s: a15 = s.read().astype(float)
nbr_pre = (a14[5] - a14[7]) / (a14[5] + a14[7])
nbr_post = (a15[5] - a15[7]) / (a15[5] + a15[7])
dnbr = nbr_pre - nbr_post
m = (np.nan_to_num(dnbr) > 0.1).astype('uint8')
gdf = gpd.GeoDataFrame(geometry=[shape(g) for g, v in features.shapes(m, mask=m == 1, transform=tr)], crs=crs)
ax = gdf.plot(color='orangered', figsize=(10, 10))
plt.savefig('pred_results/burn_scar_analysis.png')
