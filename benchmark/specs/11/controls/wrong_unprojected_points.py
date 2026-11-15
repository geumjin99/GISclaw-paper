# Wrong: samples the Albers raster with longitude/latitude (no reprojection); rasterio.sample
# returns the fill value outside the raster, so every point gets flat-ground values.
import numpy as np, geopandas as gpd, rasterio, matplotlib.pyplot as plt
src = rasterio.open('dataset/CatalinaBathymetry.tif')
z = src.read(1).astype(float)
dzdy, dzdx = np.gradient(z, src.res[1], src.res[0])
slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy))); aspect = np.degrees(np.arctan2(-dzdx, dzdy)) % 360
prof = src.profile
pts = gpd.read_file('dataset/CoralandSpongeCatalina.geojson')
with rasterio.MemoryFile() as mf:
    with mf.open(**{**prof, 'dtype': 'float64', 'count': 2}) as ds:
        ds.write(np.stack([slope, aspect]))
        vals = np.array(list(ds.sample(zip(pts.geometry.x, pts.geometry.y))))
pts['slope'] = vals[:, 0]; pts['aspect'] = vals[:, 1]
m = pts.groupby('VernacularNameCategory')[['slope', 'aspect']].mean().reset_index()
fig, ax = plt.subplots(1, 2); ax[0].bar(m.VernacularNameCategory, m.slope); ax[1].bar(m.VernacularNameCategory, m.aspect)
plt.savefig('pred_results/CoralandSponge.png')
