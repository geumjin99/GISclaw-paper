# Wrong: computes an IDW grid "for comparison", but the delivered map is the value-weighted KDE
# from the Domain Knowledge, a sampling density and not a DO surface.
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata
pts = gpd.read_file('dataset/DissolvedO2.geojson'); bay = gpd.read_file('dataset/Bay.geojson')
x, y, v = pts.geometry.x.values, pts.geometry.y.values, pts.MeasureValue.values
minx, miny, maxx, maxy = bay.total_bounds
X, Y = np.mgrid[minx:maxx:150j, miny:maxy:150j]
st = pts.assign(x=x, y=y).groupby('Station')[['x', 'y', 'MeasureValue']].mean()
idw_check = griddata(st[['x', 'y']].values, st.MeasureValue.values, (X, Y), method='nearest')
kde = gaussian_kde(np.vstack([x, y]), weights=v)
Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
fig, ax = plt.subplots(figsize=(7, 9))
ax.imshow(np.rot90(Z), extent=(minx, maxx, miny, maxy), cmap='viridis')
bay.boundary.plot(ax=ax, color='k')
fig.savefig('pred_results/interploated_water_quality.png')
