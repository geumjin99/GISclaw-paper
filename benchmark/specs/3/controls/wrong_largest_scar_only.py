# Wrong: dNBR correct, but only the single largest burned polygon is kept ("the burn scar"); the second
# (northern) scar and all other damage are dropped from the vector output and the map.
import numpy as np, rasterio, geopandas as gpd, matplotlib.pyplot as plt
from rasterio import features
from shapely.geometry import shape
def nbr(p):
    with rasterio.open(p) as s:
        a, b = s.read(5).astype(float), s.read(7).astype(float); return (a - b) / (a + b), s.transform, s.crs
n14, tr, crs = nbr('dataset/G_2014.tif'); n15, _, _ = nbr('dataset/G_2015.tif')
dnbr = n14 - n15
mask = (np.nan_to_num(dnbr) > 0.27).astype('uint8')
polys = gpd.GeoDataFrame(geometry=[shape(g) for g, v in features.shapes(mask, mask=mask == 1, transform=tr)], crs=crs)
polys['a'] = polys.area
scar = polys.nlargest(1, 'a')
fig, ax = plt.subplots(figsize=(10, 10))
scar.plot(ax=ax, color='red', alpha=0.6)
ax.set_title(f'Burn scar area: {scar.a.sum()/1e6:.1f} km2')
plt.savefig('pred_results/burn_scar_analysis.png')
