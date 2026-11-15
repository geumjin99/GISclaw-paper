# Correct: Sobel (Horn) derivatives via scipy, slope in percent, circular mean aspect,
# results kept in dictionaries.
import numpy as np, geopandas as gpd, rasterio, matplotlib.pyplot as plt
from scipy import ndimage
with rasterio.open('dataset/CatalinaBathymetry.tif') as src:
    z = src.read(1).astype(float); T = src.transform
cs = T.a
gx = ndimage.sobel(z, axis=1, mode='nearest') / (8 * cs)
gy = ndimage.sobel(z, axis=0, mode='nearest') / (8 * cs)
slope_pct = np.hypot(gx, gy) * 100
aspect = (np.degrees(np.arctan2(gx, -gy)) + 180) % 360
d = gpd.read_file('dataset/CoralandSpongeCatalina.geojson')
p = d.to_crs(3310)
cols, rows = (~T) * (p.geometry.x.values, p.geometry.y.values)
r, c = rows.astype(int), cols.astype(int)
d['slope_pct'] = slope_pct[r, c]; d['aspect_deg'] = aspect[r, c]
mean_slope = d.groupby('VernacularNameCategory')['slope_pct'].mean().to_dict()
def cm(a):
    a = np.radians(a); return np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360
mean_aspect = d.groupby('VernacularNameCategory')['aspect_deg'].apply(cm).to_dict()
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].bar(list(mean_slope), list(mean_slope.values())); ax[1].bar(list(mean_aspect), list(mean_aspect.values()))
plt.savefig('pred_results/CoralandSponge.png')
