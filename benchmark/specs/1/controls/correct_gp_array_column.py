# Correct: Gaussian-process (kriging-equivalent) interpolation inside a function; block means
# kept as a plain numpy array and passed straight to GeoDataFrame.plot(column=array).
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from shapely.geometry import box

def run():
    stations = gpd.read_file('dataset/Temperature.geojson')
    bg = gpd.read_file('dataset/block.geojson').to_crs(stations.crs)
    X = np.c_[stations.geometry.x, stations.geometry.y] / 1000.0
    y = stations['TemperatureF'].to_numpy()
    gp = GaussianProcessRegressor(C(5.0) * RBF(3.0) + WhiteKernel(0.5), normalize_y=True, random_state=0).fit(X, y)
    xmin, ymin, xmax, ymax = bg.total_bounds
    xs = np.arange(xmin, xmax, 250.0); ys = np.arange(ymin, ymax, 250.0)
    XX, YY = np.meshgrid(xs, ys)
    pred = gp.predict(np.c_[XX.ravel(), YY.ravel()] / 1000.0)
    cells = gpd.GeoDataFrame(geometry=gpd.points_from_xy(XX.ravel(), YY.ravel()), crs=bg.crs)
    hit = gpd.sjoin(cells, bg[['geometry']], predicate='within')
    sums = np.bincount(hit['index_right'].to_numpy(), weights=pred[hit.index.to_numpy()], minlength=len(bg))
    cnt = np.bincount(hit['index_right'].to_numpy(), minlength=len(bg))
    # tiny block groups without a cell centre: use the prediction at their centroid
    cpts = np.c_[bg.geometry.centroid.x, bg.geometry.centroid.y] / 1000.0
    zone_temp = np.where(cnt > 0, sums / np.maximum(cnt, 1), gp.predict(cpts))
    dens = bg['Block_Groups_Over65Density'].to_numpy()
    flag = (zone_temp >= np.percentile(zone_temp, 75)) & (dens >= np.percentile(dens, 75))
    fig, ax = plt.subplots(figsize=(9, 9))
    bg.plot(column=zone_temp, cmap='YlOrRd', legend=True, ax=ax, edgecolor='grey', linewidth=0.2)
    bg[flag].boundary.plot(ax=ax, color='navy', linewidth=1.8)
    ax.set_title('Kriged (GP) mean temperature per block group; hot & elderly-dense outlined')
    fig.savefig('pred_results/interpolated_urban_heat.png', dpi=120)

run()
