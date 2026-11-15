# Wrong: count computed correctly, but the polygon-cell map (gdf.plot of cell boxes) shows the mean
# temperature instead of the spell count.
import numpy as np, xarray as xr, geopandas as gpd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import box
da = xr.open_dataset('dataset/E1_north_america.nc').air_temperature
hot = (da > 280).values
count = np.lib.stride_tricks.sliding_window_view(hot, 5, axis=0).all(-1).sum(0)
mean_t = da.values.mean(0)
lon, lat = da.longitude.values, da.latitude.values
dx, dy = np.diff(lon).mean() / 2, np.diff(lat).mean() / 2
rows = [dict(t=float(mean_t[i, j]), geometry=box(lon[j] - dx, lat[i] - dy, lon[j] + dx, lat[i] + dy))
        for i in range(len(lat)) for j in range(len(lon))]
cells = gpd.GeoDataFrame(rows, crs='EPSG:4326')
fig, ax = plt.subplots(figsize=(10, 7))
cells.plot(column='t', cmap='Reds', legend=True, ax=ax)
plt.savefig('pred_results/temperature_statistic_vis.png')
