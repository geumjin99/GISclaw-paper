# Correct: dNBR as post - pre (burned = dNBR < -0.1), float32, polygons reprojected to WGS84.
import numpy as np, rasterio, geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape
import matplotlib.pyplot as plt

def nbr(p):
    with rasterio.open(p) as s:
        b = s.read([5, 7]).astype('float32')
        b[b >= 65535] = np.nan
        return (b[0] - b[1]) / (b[0] + b[1]), s.transform, s.crs

pre, tr, crs = nbr('dataset/G_2014.tif')
post, _, _ = nbr('dataset/G_2015.tif')
d = post - pre
mask = (d < -0.1).astype('uint8')
polys = [shape(g) for g, v in shapes(mask, mask=mask == 1, transform=tr)]
scars = gpd.GeoDataFrame(geometry=polys, crs=crs).to_crs(4326)
scars = scars[scars.to_crs(crs).area > 900 * 5]
fig, ax = plt.subplots(figsize=(9, 9))
scars.plot(ax=ax, color='firebrick')
plt.savefig('pred_results/burn_scar_analysis.png')
