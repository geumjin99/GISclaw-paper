# Wrong: slope/aspect computed on depth (elevation * -1) - slope is right, aspect points uphill.
import numpy as np, geopandas as gpd, rasterio, matplotlib.pyplot as plt
src = rasterio.open('dataset/CatalinaBathymetry.tif')
depth = -src.read(1).astype(float)
dzdy, dzdx = np.gradient(depth, src.res[1], src.res[0])
slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
aspect = np.degrees(np.arctan2(-dzdx, dzdy)) % 360
pts = gpd.read_file('dataset/CoralandSpongeCatalina.geojson').to_crs('EPSG:3310')
rc = [src.index(x, y) for x, y in zip(pts.geometry.x, pts.geometry.y)]
pts['slope'] = [slope[r, c] for r, c in rc]; pts['aspect'] = [aspect[r, c] for r, c in rc]
m = pts.groupby('VernacularNameCategory')[['slope', 'aspect']].mean().reset_index()
fig, ax = plt.subplots(1, 2); ax[0].bar(m.VernacularNameCategory, m.slope); ax[1].bar(m.VernacularNameCategory, m.aspect)
plt.savefig('pred_results/CoralandSponge.png')
