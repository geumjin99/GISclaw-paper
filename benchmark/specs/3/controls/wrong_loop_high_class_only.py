# Wrong: severity classes vectorised, but the drawing loop only iterates over the 'High' class
# (dNBR > 0.66): most of the burned core is not mapped.
import numpy as np, rasterio, geopandas as gpd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rasterio.features import shapes
from shapely.geometry import shape
def nbr(p):
    with rasterio.open(p) as s:
        a, b = s.read(5).astype(float), s.read(7).astype(float); return (a - b) / (a + b), s.transform, s.crs
n14, tr, crs = nbr('dataset/G_2014.tif'); n15, _, _ = nbr('dataset/G_2015.tif')
dnbr = n14 - n15
cls = np.digitize(np.nan_to_num(dnbr, nan=-1), [0.1, 0.27, 0.44, 0.66]).astype('int32')
recs = [(shape(g), int(v)) for g, v in shapes(cls, mask=cls > 0, transform=tr)]
gdf = gpd.GeoDataFrame({'class': [v for _, v in recs]}, geometry=[g for g, _ in recs], crs=crs)
cols = {4: '#8b008b'}
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(dnbr, cmap='RdYlBu_r', vmin=-0.3, vmax=0.7,
          extent=[tr.c, tr.c + tr.a * dnbr.shape[1], tr.f + tr.e * dnbr.shape[0], tr.f])
for k, c in cols.items():
    sub = gdf[gdf['class'] == k]
    sub.plot(ax=ax, facecolor=c, edgecolor=c, alpha=0.75)
ax.set_title('Burn scars (high severity)')
plt.savefig('pred_results/burn_scar_analysis.png')
