# Wrong: uses abs(DEM) instead of inverting: land/coastal cells get a positive "depth" and a speed.
import numpy as np, pandas as pd, geopandas as gpd, rasterio, matplotlib.pyplot as plt
src = rasterio.open('dataset/AtlanticDEM.tif')
dem = src.read(1, masked=True).filled(np.nan).astype(float)
speed = np.sqrt(9.81 * np.abs(dem))
paths = gpd.read_file('dataset/TsuniamiPaths.gpkg').set_crs(src.crs.to_wkt(), allow_override=True)
inv = ~src.transform
rows = []
for _, r in paths.iterrows():
    line = r.geometry.geoms[0]
    d = np.arange(0, line.length, 1000.0); d = np.append(d, line.length)
    mids = [line.interpolate((a + b) / 2) for a, b in zip(d[:-1], d[1:])]
    cc, rr = inv * (np.array([p.x for p in mids]), np.array([p.y for p in mids]))
    sp = speed[rr.astype(int), cc.astype(int)]
    ok = np.isfinite(sp) & (sp > 0)
    rows.append({'route': r.Route, 'hours': (np.diff(d)[ok] / sp[ok]).sum() / 3600})
tt = pd.DataFrame(rows); print(tt); tt.to_csv('pred_results/travel_time.csv', index=False)
plt.imshow(speed, cmap='viridis'); plt.colorbar(); plt.savefig('pred_results/tsunami_speed_map.png')
