# Correct: Layer 'S' station means, ordinary kriging on a 1 km grid masked to the Bay, drawn with
# imshow in KILOMETRES (extent = metre coordinates / 1000, axes "Easting (km)").
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
import shapely
from pykrige.ok import OrdinaryKriging
pts = gpd.read_file('dataset/DissolvedO2.geojson'); bay = gpd.read_file('dataset/Bay.geojson').to_crs(pts.crs)
surf = pts[(pts.Layer.str.strip() == 'S') & pts.MeasureValue.notna()]
st = surf.groupby('Station').agg(x=('geometry', lambda g: g.iloc[0].x), y=('geometry', lambda g: g.iloc[0].y),
                                 do=('MeasureValue', 'mean'))
minx, miny, maxx, maxy = bay.total_bounds
gx = np.arange(minx, maxx + 1000, 1000); gy = np.arange(miny, maxy + 1000, 1000)
GX, GY = np.meshgrid(gx, gy)
inside = shapely.contains_xy(bay.geometry.union_all(), GX, GY)
ok = OrdinaryKriging(st.x.values / 1000, st.y.values / 1000, st['do'].values, variogram_model='spherical')
zg, var = ok.execute('grid', gx / 1000.0, gy / 1000.0)
zg = np.where(inside, np.asarray(zg), np.nan)
ext = [gx[0] / 1000, gx[-1] / 1000, gy[0] / 1000, gy[-1] / 1000]
fig, ax = plt.subplots(figsize=(7, 10))
im = ax.imshow(zg, origin='lower', extent=ext, cmap='viridis')
ax.scatter(st.x / 1000, st.y / 1000, c=st['do'], s=12, edgecolors='white', linewidths=0.4)
ax.set_xlabel('Easting (km, UTM 18N)'); ax.set_ylabel('Northing (km)')
plt.colorbar(im, ax=ax, label='DO (mg/L)')
plt.savefig('pred_results/interploated_water_quality.png', dpi=100)
