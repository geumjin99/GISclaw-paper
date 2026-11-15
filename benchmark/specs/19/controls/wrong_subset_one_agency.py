# Wrong: only one agency's stations are interpolated (silent filter) and extrapolated across the whole Bay.
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from scipy.interpolate import griddata
pts = gpd.read_file('dataset/DissolvedO2.geojson'); bay = gpd.read_file('dataset/Bay.geojson')
print(pts.Agency.value_counts())
pts = pts[pts.Agency == pts.Agency.value_counts().index[-1]]
pts['x'] = pts.geometry.x; pts['y'] = pts.geometry.y
st = pts.groupby('Station')[['x', 'y', 'MeasureValue']].mean()
minx, miny, maxx, maxy = bay.total_bounds
X, Y = np.meshgrid(np.linspace(minx, maxx, 200), np.linspace(maxy, miny, 300))
Z = griddata(st[['x', 'y']].values, st.MeasureValue.values, (X, Y), method='nearest')
inside = gpd.points_from_xy(X.ravel(), Y.ravel()).within(bay.geometry.union_all()).reshape(X.shape)
Z[~inside] = np.nan
plt.imshow(Z, extent=(minx, maxx, miny, maxy), cmap='RdYlBu'); plt.colorbar()
plt.savefig('pred_results/interploated_water_quality.png')
