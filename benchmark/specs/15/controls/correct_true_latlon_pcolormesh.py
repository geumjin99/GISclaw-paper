# Correct: true lon/lat computed from the rotated-pole grid with cartopy, masked TEC drawn with
# contourf on an orthographic map with land and coastlines.
import numpy as np, xarray as xr, matplotlib.pyplot as plt, cartopy.crs as ccrs, cartopy.feature as cf
ds = xr.open_dataset('dataset/space_weather.nc')
pole = ds.rotated_pole.attrs
src = ccrs.RotatedPole(pole_longitude=pole['grid_north_pole_longitude'], pole_latitude=pole['grid_north_pole_latitude'])
rl, rt = np.meshgrid(ds.rLon.values, ds.rLat.values)
xyz = ccrs.PlateCarree().transform_points(src, rl, rt)
lon2d, lat2d = xyz[..., 0], xyz[..., 1]
tec = np.where(ds.TEC.values < 0, np.nan, ds.TEC.values)
ax = plt.axes(projection=ccrs.Orthographic(0, 45))
ax.add_feature(cf.LAND, color='0.9'); ax.add_feature(cf.COASTLINE)
m = ax.contourf(lon2d, lat2d, tec, 40, transform=ccrs.PlateCarree())
plt.colorbar(m); plt.savefig('pred_results/TEC_vis.png')
