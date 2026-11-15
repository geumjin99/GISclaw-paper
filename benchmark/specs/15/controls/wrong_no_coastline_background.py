# Wrong: TEC correctly placed in the rotated-pole CRS, but no coastline / geographic background
# although the task text asks for one.
import xarray as xr, matplotlib.pyplot as plt, cartopy.crs as ccrs, numpy.ma as ma
ds = xr.open_dataset('dataset/space_weather.nc')
rp = ccrs.RotatedPole(pole_longitude=180, pole_latitude=45)
tec = ma.masked_less(ds.TEC.values, 0)
ax = plt.axes(projection=rp)
cs = ax.contourf(ds.rLon, ds.rLat, tec, 50, transform=rp)
plt.colorbar(cs)
plt.savefig('pred_results/TEC_vis.png')
