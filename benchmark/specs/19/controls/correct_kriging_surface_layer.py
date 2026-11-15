# Correct: surface-layer (Layer 'S') station means, ordinary kriging (exponential variogram),
# mapped as a point grid coloured by the predicted DO.
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
pts = gpd.read_file('dataset/DissolvedO2.geojson'); bay = gpd.read_file('dataset/Bay.geojson')
surf = pts[pts.Layer.str.strip() == 'S']
st = surf.groupby('Station').agg(x=('geometry', lambda g: g.iloc[0].x), y=('geometry', lambda g: g.iloc[0].y), do=('MeasureValue', 'mean'))
ok = OrdinaryKriging(st.x.values, st.y.values, st.do.values, variogram_model='exponential')
minx, miny, maxx, maxy = bay.total_bounds
gx = np.linspace(minx, maxx, 120); gy = np.linspace(miny, maxy, 220)
z, ss = ok.execute('grid', gx, gy)
grid = gpd.GeoDataFrame({'do_pred': np.asarray(z).ravel()}, geometry=gpd.points_from_xy(*[a.ravel() for a in np.meshgrid(gx, gy)]), crs=pts.crs)
grid = grid[grid.within(bay.geometry.union_all())]
ax = grid.plot(column='do_pred', markersize=3, legend=True, cmap='viridis'); bay.boundary.plot(ax=ax, color='k')
plt.savefig('pred_results/interploated_water_quality.png')
