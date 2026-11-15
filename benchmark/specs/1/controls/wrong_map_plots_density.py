# Wrong: computes the kriged block means correctly, but the delivered map shows elderly density, not temperature.
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
t = gpd.read_file('dataset/Temperature.geojson'); b = gpd.read_file('dataset/block.geojson')
ok = OrdinaryKriging(t.geometry.x.values, t.geometry.y.values, t.TemperatureF.values, variogram_model='spherical')
xmin, ymin, xmax, ymax = b.total_bounds
gx = np.linspace(xmin, xmax, 120); gy = np.linspace(ymin, ymax, 120)
z, _ = ok.execute('grid', gx, gy)
GX, GY = np.meshgrid(gx, gy)
pts = gpd.GeoDataFrame({'temp': np.asarray(z).ravel()}, geometry=gpd.points_from_xy(GX.ravel(), GY.ravel()), crs=b.crs)
j = gpd.sjoin(pts, b[['OBJECTID', 'geometry']], predicate='within')
means = j.groupby('OBJECTID').temp.mean().rename('mean_temp').reset_index()
blocks = b.merge(means, on='OBJECTID', how='left')
fig, ax = plt.subplots(figsize=(9, 9))
blocks.plot(column='Block_Groups_Over65Density', cmap='OrRd', legend=True, ax=ax)
hot = blocks[(blocks.mean_temp > blocks.mean_temp.quantile(.75)) & (blocks.Block_Groups_Over65Density > blocks.Block_Groups_Over65Density.quantile(.75))]
hot.boundary.plot(ax=ax, color='blue', linewidth=2)
ax.set_title('Urban heat & elderly density')
plt.savefig('pred_results/interpolated_urban_heat.png')
