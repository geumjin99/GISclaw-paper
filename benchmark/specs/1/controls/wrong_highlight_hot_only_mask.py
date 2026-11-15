# Wrong: highlights all hot blocks (the elderly mask is computed but not used).
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
stations = gpd.read_file('dataset/Temperature.geojson')
bg = gpd.read_file('dataset/block.geojson').to_crs(stations.crs)
X = np.c_[stations.geometry.x, stations.geometry.y] / 1000.0
gp = GaussianProcessRegressor(C(5.0) * RBF(3.0) + WhiteKernel(0.5), normalize_y=True, random_state=0).fit(X, stations['TemperatureF'].to_numpy())
xmin, ymin, xmax, ymax = bg.total_bounds
XX, YY = np.meshgrid(np.arange(xmin, xmax, 250.0), np.arange(ymin, ymax, 250.0))
pred = gp.predict(np.c_[XX.ravel(), YY.ravel()] / 1000.0)
cells = gpd.GeoDataFrame({'t': pred}, geometry=gpd.points_from_xy(XX.ravel(), YY.ravel()), crs=bg.crs)
hit = gpd.sjoin(cells, bg[['geometry']], predicate='within')
bg['AvgTempF'] = hit.groupby('index_right')['t'].mean()
cpts = np.c_[bg.geometry.centroid.x, bg.geometry.centroid.y] / 1000.0
bg['AvgTempF'] = bg['AvgTempF'].fillna(dict(zip(bg.index, gp.predict(cpts))))
thr_t = bg['AvgTempF'].quantile(0.75)
thr_e = bg['Block_Groups_Over65Density'].quantile(0.75)
hot = bg['AvgTempF'] > thr_t
eld = bg['Block_Groups_Over65Density'] > thr_e
fig, ax = plt.subplots(figsize=(9, 9))
bg.plot(column='AvgTempF', cmap='YlOrRd', legend=True, ax=ax, edgecolor='grey', linewidth=0.2)
bg[hot].plot(ax=ax, facecolor='none', edgecolor='crimson', linewidth=2.0)
fig.savefig('pred_results/interpolated_urban_heat.png', dpi=120)
