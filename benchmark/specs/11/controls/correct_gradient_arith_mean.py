# Correct: np.gradient with the true cell size, arithmetic mean of aspect, rasterio.sample.
import numpy as np, geopandas as gpd, rasterio, matplotlib.pyplot as plt
src = rasterio.open('dataset/CatalinaBathymetry.tif')
z = src.read(1).astype(float)
dzdy, dzdx = np.gradient(z, src.res[1], src.res[0])     # dzdy along rows (southwards)
slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
aspect = np.degrees(np.arctan2(-dzdx, dzdy)) % 360       # downslope azimuth
pts = gpd.read_file('dataset/CoralandSpongeCatalina.geojson').to_crs('EPSG:3310')
rc = [src.index(x, y) for x, y in zip(pts.geometry.x, pts.geometry.y)]
pts['slope'] = [slope[r, c] for r, c in rc]
pts['aspect'] = [aspect[r, c] for r, c in rc]
m = pts.groupby('VernacularNameCategory')[['slope', 'aspect']].mean().reset_index()
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].bar(m.VernacularNameCategory, m.slope); ax[1].bar(m.VernacularNameCategory, m.aspect)
for a in ax: a.tick_params(axis='x', rotation=90)
plt.tight_layout(); plt.savefig('pred_results/CoralandSponge.png')
