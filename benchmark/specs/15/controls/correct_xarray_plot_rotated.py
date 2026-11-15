# Correct: xarray masking with where() and DataArray.plot.contourf in the rotated-pole CRS on a
# Robinson map.
import xarray as xr, matplotlib.pyplot as plt, cartopy.crs as ccrs
ds = xr.open_dataset('dataset/space_weather.nc')
tec = ds.TEC.where(ds.TEC >= 0)
fig = plt.figure(figsize=(9, 7))
ax = plt.axes(projection=ccrs.Robinson())
tec.plot.contourf(ax=ax, transform=ccrs.RotatedPole(pole_longitude=180, pole_latitude=45), levels=30, cmap='viridis')
ax.coastlines(); ax.set_global()
plt.savefig('pred_results/TEC_vis.png')
