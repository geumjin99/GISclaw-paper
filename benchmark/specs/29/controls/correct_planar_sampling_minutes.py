# Correct: planar (Albers) straight paths sampled every 2 km with rasterio.sample,
# float speed (no Int), times reported in minutes in a CSV; simple matplotlib map.
import numpy as np, pandas as pd, geopandas as gpd, rasterio, matplotlib.pyplot as plt
src = rasterio.open('dataset/AtlanticDEM.tif')
paths = gpd.read_file('dataset/TsuniamiPaths.gpkg').set_crs(src.crs.to_wkt(), allow_override=True)
res = []
for _, r in paths.iterrows():
    line = r.geometry.geoms[0]
    n = int(line.length // 2000) + 1
    d = np.linspace(0, line.length, n + 1)
    mids = [line.interpolate((a + b) / 2) for a, b in zip(d[:-1], d[1:])]
    dem = np.array([v[0] for v in src.sample([(p.x, p.y) for p in mids])], float)
    sp = np.sqrt(9.80665 * np.clip(-dem, 0, None))
    ok = sp > 0
    minutes = (np.diff(d)[ok] / sp[ok]).sum() / 60
    res.append((r.Route, minutes))
df = pd.DataFrame(res, columns=['route', 'travel_minutes'])
df.to_csv('pred_results/travel_times_minutes.csv', index=False)
ax = paths.plot(color='red'); ax.set_title('Tsunami paths'); plt.savefig('pred_results/tsunami_map.png')
