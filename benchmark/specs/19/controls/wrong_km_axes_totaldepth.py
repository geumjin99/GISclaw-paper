# Wrong: kriging map drawn with imshow on km axes, but the variable interpolated is TotalDepth,
# not dissolved oxygen.
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
import shapely
from pykrige.ok import OrdinaryKriging
pts = gpd.read_file('dataset/DissolvedO2.geojson'); bay = gpd.read_file('dataset/Bay.geojson').to_crs(pts.crs)
st = pts.groupby('Station').agg(x=('geometry', lambda g: g.iloc[0].x), y=('geometry', lambda g: g.iloc[0].y),
                                v=('TotalDepth', 'mean'))
st = st[np.isfinite(st.v)]
minx, miny, maxx, maxy = bay.total_bounds
gx = np.arange(minx, maxx + 1000, 1000); gy = np.arange(miny, maxy + 1000, 1000)
GX, GY = np.meshgrid(gx, gy)
inside = shapely.contains_xy(bay.geometry.union_all(), GX, GY)
ok = OrdinaryKriging(st.x.values / 1000, st.y.values / 1000, st.v.values, variogram_model='spherical')
zg, _ = ok.execute('grid', gx / 1000.0, gy / 1000.0)
zg = np.where(inside, np.asarray(zg), np.nan)
fig, ax = plt.subplots(figsize=(7, 10))
im = ax.imshow(zg, origin='lower', extent=[gx[0] / 1000, gx[-1] / 1000, gy[0] / 1000, gy[-1] / 1000], cmap='viridis')
ax.set_xlabel('Easting (km)'); plt.colorbar(im, ax=ax)
plt.savefig('pred_results/interploated_water_quality.png', dpi=100)
