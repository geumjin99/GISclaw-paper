# Wrong: speed = sqrt(depth) - drops g from the shallow-water formula.
import numpy as np, pandas as pd, geopandas as gpd, rasterio, matplotlib.pyplot as plt
src = rasterio.open('dataset/AtlanticDEM.tif')
paths = gpd.read_file('dataset/TsuniamiPaths.gpkg').set_crs(src.crs.to_wkt(), allow_override=True)
rows = []
for _, r in paths.iterrows():
    line = r.geometry.geoms[0]; d = np.linspace(0, line.length, 3001)
    pts = [line.interpolate(x) for x in (d[:-1] + d[1:]) / 2]
    dem = np.array([v[0] for v in src.sample([(q.x, q.y) for q in pts])], float)
    sp = np.sqrt(np.clip(-dem, 0, None)); ok = sp > 0
    rows.append({'Route': r.Route, 'hours': (np.diff(d)[ok] / sp[ok]).sum() / 3600})
tt = pd.DataFrame(rows); tt.to_csv('pred_results/travel_time.csv', index=False)
paths.plot(); plt.savefig('pred_results/tsunami.png')
