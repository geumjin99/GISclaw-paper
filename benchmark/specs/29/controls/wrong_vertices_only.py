# Wrong: no densification - speed sampled only at the path's own vertices (start and end), time = length / mean vertex speed.
import numpy as np, pandas as pd, geopandas as gpd, rasterio, matplotlib.pyplot as plt
src = rasterio.open('dataset/AtlanticDEM.tif')
paths = gpd.read_file('dataset/TsuniamiPaths.gpkg').set_crs(src.crs.to_wkt(), allow_override=True)
rows = []
for _, r in paths.iterrows():
    line = r.geometry.geoms[0]
    dem = np.array([v[0] for v in src.sample(list(line.coords))], float)
    sp = np.sqrt(9.80665 * np.clip(-dem, 0, None))
    rows.append({'Route': r.Route, 'TravelTime_h': line.length / sp[sp > 0].mean() / 3600})
tt = pd.DataFrame(rows); print(tt); tt.to_csv('pred_results/travel_time.csv', index=False)
ax = paths.plot(color='r'); plt.savefig('pred_results/tsunami_paths.png')
