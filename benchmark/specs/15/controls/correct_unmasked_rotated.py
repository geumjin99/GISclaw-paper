# Correct: raw TEC with its negative cells drawn (the task text does not ask for masking),
# correctly placed in the rotated-pole CRS.
import xarray as xr, matplotlib.pyplot as plt, cartopy.crs as ccrs
ds = xr.open_dataset('dataset/space_weather.nc')
rp = ccrs.RotatedPole(pole_longitude=180, pole_latitude=45)
tec = ds.TEC.values                                   # negative (invalid) values kept
ax = plt.axes(projection=rp)
ax.contourf(ds.rLon, ds.rLat, tec, 50, transform=rp); ax.coastlines()
plt.savefig('pred_results/TEC_vis.png')
