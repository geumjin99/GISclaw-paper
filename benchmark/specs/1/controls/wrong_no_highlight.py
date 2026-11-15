# Wrong: kriged choropleth only; the required highlight of hot & elderly-dense block groups is missing entirely,
# and the demographic layer is not used for anything.
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
t = gpd.read_file('dataset/Temperature.geojson'); b = gpd.read_file('dataset/block.geojson')
ok = OrdinaryKriging(t.geometry.x.values, t.geometry.y.values, t.TemperatureF.values, variogram_model='spherical')
gx = np.linspace(*b.total_bounds[[0, 2]], 100); gy = np.linspace(*b.total_bounds[[1, 3]], 100)
z, _ = ok.execute('grid', gx, gy)
GX, GY = np.meshgrid(gx, gy)
pts = gpd.GeoDataFrame({'t': np.asarray(z).ravel()}, geometry=gpd.points_from_xy(GX.ravel(), GY.ravel()), crs=b.crs)
j = gpd.sjoin(pts, b, predicate='within')
b['t_mean'] = j.groupby('index_right').t.mean()
b.plot(column='t_mean', cmap='OrRd', legend=True, figsize=(9, 9))
plt.savefig('pred_results/interpolated_urban_heat.png')
