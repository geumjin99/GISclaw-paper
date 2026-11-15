# Wrong: block-group mean temperatures are interpolated (IDW), the choropleth is drawn -
# but saved to another PNG; interpolated_urban_heat.png shows only the hot & elderly highlight
# over plain block outlines (no temperature choropleth in the requested map).
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from rasterstats import zonal_stats
from rasterio.transform import from_origin
t = gpd.read_file('dataset/Temperature.geojson'); b = gpd.read_file('dataset/block.geojson')
x, y, z = t.geometry.x.values, t.geometry.y.values, t['TemperatureF'].values
xmin, ymin, xmax, ymax = b.total_bounds; res = 100.0
gx = np.arange(xmin, xmax, res); gy = np.arange(ymax, ymin, -res)
GX, GY = np.meshgrid(gx, gy)
d = np.sqrt((GX[..., None] - x) ** 2 + (GY[..., None] - y) ** 2) + 1e-6
w = 1 / d ** 2
surf = (w * z).sum(-1) / w.sum(-1)
zs = zonal_stats(b, surf, affine=from_origin(xmin, ymax, res, res), stats=['mean'])
b['idw_temp'] = [s['mean'] for s in zs]
ax = b.plot(column='idw_temp', cmap='OrRd', legend=True)
plt.savefig('pred_results/temperature_by_block.png')
plt.close()
hot = b[(b.idw_temp > b.idw_temp.quantile(.75)) & (b.Block_Groups_Over65Density > b.Block_Groups_Over65Density.quantile(.75))]
ax = b.boundary.plot(color='grey', linewidth=0.3)
hot.plot(ax=ax, color='red')
plt.savefig('pred_results/interpolated_urban_heat.png')
