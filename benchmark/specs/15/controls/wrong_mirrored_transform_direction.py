# Wrong: transform_points called in the wrong direction (geographic -> rotated), so the map is
# mirrored north-south and drawn at the wrong place.
import numpy as np, matplotlib.pyplot as plt, cartopy.crs as ccrs, cartopy.feature as cfeature
import xarray as xr
ds = xr.open_dataset('dataset/space_weather.nc')
tec = ds['TEC']
rlat = ds['rLat'].values
rlon = ds['rLon'].values
rlon2d, rlat2d = np.meshgrid(rlon, rlat)
rotated_crs = ccrs.RotatedPole(pole_longitude=180.0, pole_latitude=45.0)
geom_crs = ccrs.PlateCarree()
points_rot = np.column_stack([rlon2d.ravel(), rlat2d.ravel()])
lonlat = rotated_crs.transform_points(geom_crs, points_rot[:,0], points_rot[:,1])
lon = lonlat[:,0].reshape(rlon2d.shape)
lat = lonlat[:,1].reshape(rlat2d.shape)
fig = plt.figure(figsize=(8,5))
ax = plt.axes(projection=geom_crs)
margin = 2
ax.set_extent([np.nanmin(lon)-margin, np.nanmax(lon)+margin,
               np.nanmin(lat)-margin, np.nanmax(lat)+margin], crs=geom_crs)
clevels = np.linspace(np.nanmin(tec.values), np.nanmax(tec.values), 15)
cf = ax.contourf(lon, lat, tec.values, levels=clevels, transform=geom_crs, cmap='viridis', extend='both')
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
cbar.set_label('TEC (1E16 e/m^2)')
ax.set_title('Total Electron Content (TEC)')
plt.savefig('pred_results/TEC_vis.png', dpi=300, bbox_inches='tight')
plt.close()
