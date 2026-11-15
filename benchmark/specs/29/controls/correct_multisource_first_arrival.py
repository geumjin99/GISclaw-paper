# Correct: one combined first-arrival map for all three tsunami origins (cell-wise minimum of the
# three least-time surfaces), saved in minutes; no per-route table.
import numpy as np, rasterio, geopandas as gpd, matplotlib.pyplot as plt
from skimage.graph import MCP_Geometric
with rasterio.open('dataset/AtlanticDEM.tif') as s:
    dem = s.read(1, masked=True).filled(np.nan); prof = s.profile; T = s.transform
cell = abs(T.a)
v = np.sqrt(9.81 * np.where(dem < 0, -dem, np.nan))
cost = np.where(np.isfinite(v) & (v > 0), cell / np.where(np.isfinite(v) & (v > 0), v, 1), np.inf)
paths = gpd.read_file('dataset/TsuniamiPaths.gpkg')
def _least_time(geom):   # single-source surfaces stay local: only the combined map is kept
    ln = geom.geoms[0] if geom.geom_type == 'MultiLineString' else geom
    c, r = ~T * ln.coords[0]
    cum, _ = MCP_Geometric(cost).find_costs([(int(r), int(c))])
    cum[~np.isfinite(cost)] = np.nan
    return cum
first_arrival_min = np.fmin.reduce([_least_time(g) for g in paths.geometry]) / 60.0
prof.update(dtype='float32', nodata=np.nan, count=1)
with rasterio.open('pred_results/first_arrival_minutes.tif', 'w', **prof) as d:
    d.write(first_arrival_min.astype('float32'), 1)
plt.imshow(first_arrival_min / 60, cmap='magma_r'); plt.colorbar(label='first arrival (h)')
plt.title('Tsunami first arrival, all sources'); plt.savefig('pred_results/first_arrival.png')
