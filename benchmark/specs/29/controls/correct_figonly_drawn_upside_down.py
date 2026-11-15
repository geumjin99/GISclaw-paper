# Correct: the analysis is right (least-time arrival surface, GeoTIFF in seconds); only the presentation is wrong
# (map drawn upside down, origin='lower'). The optional north-up figure check fails.
import numpy as np, rasterio, geopandas as gpd, matplotlib.pyplot as plt
from skimage.graph import MCP_Geometric
with rasterio.open('dataset/AtlanticDEM.tif') as s:
    dem = s.read(1, masked=True).filled(np.nan); prof = s.profile; T = s.transform
v = np.sqrt(9.80665 * np.where(dem < 0, -dem, np.nan))
cost = np.where(v > 0.5, 5000.0 / np.where(v > 0.5, v, 1), np.inf)
src = gpd.read_file('dataset/TsunamiSources.gpkg')
p = src[src.Name == 'Puerto Rico Trench'].geometry.iloc[0]
c, r = ~T * (p.x, p.y)
arrival_s, _ = MCP_Geometric(cost).find_costs([(int(r), int(c))])
arrival_s[~np.isfinite(cost)] = np.nan
prof.update(dtype='float32', nodata=np.nan, count=1)
with rasterio.open('pred_results/arrival_time_seconds.tif', 'w', **prof) as d:
    d.write(arrival_s.astype('float32'), 1)
plt.imshow(arrival_s / 3600, cmap='magma_r', origin='lower'); plt.colorbar(label='h'); plt.savefig('pred_results/arrival.png')
