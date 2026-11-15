# Wrong: interpolates the station depth (TotalDepth) instead of dissolved oxygen (MeasureValue).
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from scipy.interpolate import griddata
pts = gpd.read_file('dataset/DissolvedO2.geojson'); bay = gpd.read_file('dataset/Bay.geojson')
st = pts.groupby('Station').agg(x=('geometry', lambda g: g.iloc[0].x), y=('geometry', lambda g: g.iloc[0].y), v=('TotalDepth', 'mean'))
minx, miny, maxx, maxy = bay.total_bounds
X, Y = np.meshgrid(np.linspace(minx, maxx, 200), np.linspace(maxy, miny, 300))
Z = griddata(st[['x', 'y']].values, st.v.values, (X, Y), method='linear')   # interpolates station depth, not DO
plt.imshow(Z, extent=(minx, maxx, miny, maxy)); plt.colorbar(); plt.savefig('pred_results/interploated_water_quality.png')
