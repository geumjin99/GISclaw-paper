# Correct (reciprocity, as in NOAA reverse TTT maps): least-time surface computed FROM Virginia Beach;
# its value at each source = travel time source -> Virginia Beach; times written to CSV, map of the surface.
import numpy as np, pandas as pd, geopandas as gpd, rasterio, matplotlib.pyplot as plt
from skimage.graph import MCP_Geometric
src = rasterio.open('dataset/AtlanticDEM.tif')
dem = src.read(1, masked=True).filled(np.nan).astype(float)
speed = np.where(dem < 0, np.sqrt(9.80665 * -np.where(dem < 0, dem, 0)), np.nan)
cost = np.where(np.isfinite(speed) & (speed > 0), src.res[0] / np.where(speed > 0, speed, 1), np.inf)
pts = gpd.read_file('dataset/TsunamiSources.gpkg').set_crs(src.crs.to_wkt(), allow_override=True)
inv = ~src.transform
def rc(p):
    c, r = inv * (p.x, p.y); return int(r), int(c)
vb = rc(pts[pts.Name == 'Virginia Beach'].geometry.iloc[0])
# nearest ocean cell to Virginia Beach
oc = np.argwhere(np.isfinite(cost)); vb = tuple(oc[np.argmin(((oc - np.array(vb)) ** 2).sum(1))])
ttt_hours = MCP_Geometric(cost).find_costs([vb])[0] / 3600
ttt_hours[~np.isfinite(cost)] = np.nan
rows = []
for _, s in pts[pts.Name != 'Virginia Beach'].iterrows():
    r, c = rc(s.geometry)
    win = ttt_hours[max(r - 2, 0):r + 3, max(c - 2, 0):c + 3]
    rows.append({'source': s.Name, 'travel_time_to_virginia_beach_h': float(np.nanmin(win))})
tt = pd.DataFrame(rows); print(tt); tt.to_csv('pred_results/tsunami_travel_times.csv', index=False)
fig, ax = plt.subplots(figsize=(12, 7)); im = ax.imshow(ttt_hours, cmap='magma_r'); fig.colorbar(im, label='hours to Virginia Beach')
fig.savefig('pred_results/tsunami_arrival_map.png')
