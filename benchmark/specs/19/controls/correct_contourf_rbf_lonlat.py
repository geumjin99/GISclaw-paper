# Correct: RBF (thin-plate) interpolation of station means on a lon/lat grid, mapped with contourf
# in geographic coordinates (the drawn field is reconstructed from the contourf call).
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
pts = gpd.read_file('dataset/DissolvedO2.geojson').to_crs(4326)
bay = gpd.read_file('dataset/Bay.geojson').to_crs(4326)
pts['lon'] = pts.geometry.x; pts['lat'] = pts.geometry.y
st = pts.groupby('Station')[['lon', 'lat', 'MeasureValue']].mean()
minx, miny, maxx, maxy = bay.total_bounds
lon = np.linspace(minx, maxx, 160); lat = np.linspace(miny, maxy, 300)
LON, LAT = np.meshgrid(lon, lat)
rbf = RBFInterpolator(st[['lon', 'lat']].values, st.MeasureValue.values, kernel='thin_plate_spline', smoothing=1.0)
do = rbf(np.c_[LON.ravel(), LAT.ravel()]).reshape(LON.shape)
inside = gpd.points_from_xy(LON.ravel(), LAT.ravel()).within(bay.union_all()).reshape(LON.shape)
do_masked = np.where(inside, do, np.nan)
fig, ax = plt.subplots(figsize=(7, 9))
cs = ax.contourf(LON, LAT, do_masked, levels=15, cmap='RdYlBu')
bay.boundary.plot(ax=ax, color='k', linewidth=0.5)
fig.colorbar(cs, label='DO (mg/L)')
fig.savefig('pred_results/interploated_water_quality.png')
