# Wrong: correct dNBR, but contour polygons are left in pixel (row/col) coordinates while the
# layer is labelled with the image CRS, so the "burn scars" sit at the wrong place.
import numpy as np, rasterio, geopandas as gpd
from shapely.geometry import Polygon
from skimage import measure
import matplotlib.pyplot as plt

def nbr(p):
    with rasterio.open(p) as s:
        return (s.read(5).astype(float) - s.read(7)) / (s.read(5).astype(float) + s.read(7)), s.crs
a, crs = nbr('dataset/G_2014.tif')
b, _ = nbr('dataset/G_2015.tif')
dnbr = a - b
polys = [Polygon(c[:, ::-1]) for c in measure.find_contours((dnbr > 0.27).astype(float), 0.5) if len(c) > 3]
scars = gpd.GeoDataFrame(geometry=polys, crs=crs)
fig, ax = plt.subplots()
ax.imshow(dnbr, cmap='RdYlGn_r')
scars.boundary.plot(ax=ax, color='k')
plt.savefig('pred_results/burn_scar_analysis.png')
