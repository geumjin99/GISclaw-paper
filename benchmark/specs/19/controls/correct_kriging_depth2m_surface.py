# Correct: near-surface water defined by depth (Depth <= 2 m: all 131 stations, Layer codes
# M/VP/S/BS...) instead of the Layer == 'S' flag; station means, ordinary kriging (spherical) on
# a 1.5 km grid masked to the Bay, imshow.
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from shapely import contains_xy
from pykrige.ok import OrdinaryKriging
pts = gpd.read_file('dataset/DissolvedO2.geojson'); bay = gpd.read_file('dataset/Bay.geojson').to_crs(pts.crs)
s = pts[(pts.Depth <= 2) & pts.MeasureValue.notna()].copy()
s['x'] = s.geometry.x; s['y'] = s.geometry.y
st = s.groupby(['x', 'y'], as_index=False).agg(do=('MeasureValue', 'mean'))
minx, miny, maxx, maxy = bay.total_bounds
cell = 1500.0
xs = np.arange(minx + cell / 2, maxx, cell); ys = np.arange(miny + cell / 2, maxy, cell)
GX, GY = np.meshgrid(xs, ys)
mask = contains_xy(bay.geometry.union_all(), GX, GY)
ok = OrdinaryKriging(st.x.values, st.y.values, st['do'].values, variogram_model='spherical', nlags=12)
zz, _ = ok.execute('points', GX[mask], GY[mask])
Z = np.full(GX.shape, np.nan); Z[mask] = np.asarray(zz)
fig, ax = plt.subplots(figsize=(7, 10))
im = ax.imshow(Z, origin='lower', extent=[minx, maxx, miny, maxy], cmap='RdYlBu')
bay.boundary.plot(ax=ax, color='k', lw=0.5)
plt.colorbar(im, ax=ax, label='DO (mg/L), depth <= 2 m')
plt.savefig('pred_results/interploated_water_quality.png', dpi=100)
