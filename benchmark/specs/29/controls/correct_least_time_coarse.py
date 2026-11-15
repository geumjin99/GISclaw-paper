# Correct: least-time (fastest path) arrival surfaces from each source on a 10 km grid (2x2 block mean of speed);
# arrival time at Virginia Beach tabulated per route; map of the La Palma surface.
import numpy as np, pandas as pd, geopandas as gpd, rasterio, matplotlib.pyplot as plt
from skimage.graph import MCP_Geometric
src = rasterio.open('dataset/AtlanticDEM.tif')
dem = src.read(1, masked=True).filled(np.nan).astype(float)
h, w = (dem.shape[0] // 2) * 2, (dem.shape[1] // 2) * 2
d2 = dem[:h, :w].reshape(h // 2, 2, w // 2, 2)
depth = -np.nanmean(d2, axis=(1, 3))
speed = np.where(depth > 0, np.sqrt(9.80665 * np.clip(depth, 0, None)), np.nan)
cost = np.where(np.isfinite(speed), 10000.0 / np.where(np.isfinite(speed), speed, 1), np.inf)
paths = gpd.read_file('dataset/TsuniamiPaths.gpkg').set_crs(src.crs.to_wkt(), allow_override=True)
inv = ~src.transform
def rc(x, y):
    c, r = inv * (x, y); return int(r) // 2, int(c) // 2
rows, surf = [], {}
for _, p in paths.iterrows():
    line = p.geometry.geoms[0]
    s, e = rc(*line.coords[0]), rc(*line.coords[-1])
    t = MCP_Geometric(cost).find_costs([s])[0] / 3600
    t[~np.isfinite(cost)] = np.nan
    surf[p.Route] = t
    win = t[e[0] - 2:e[0] + 3, e[1] - 2:e[1] + 3]
    rows.append({'Route': p.Route, 'arrival_h': float(np.nanmin(win))})
tt = pd.DataFrame(rows); print(tt); tt.to_csv('pred_results/arrival_times.csv', index=False)
lp = surf['La Palma to Virginia Beach']
fig, ax = plt.subplots(figsize=(12, 7)); im = ax.imshow(lp, cmap='magma_r'); fig.colorbar(im, label='hours')
fig.savefig('pred_results/arrival_time_map.png')
