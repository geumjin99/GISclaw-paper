# Correct: true lon/lat derived by rotating the grid 45 deg by hand, raw (unmasked) TEC drawn
# with contourf on a PlateCarree map with land, ocean and coastlines.
import numpy as np, xarray as xr, matplotlib.pyplot as plt, cartopy.crs as ccrs, cartopy.feature as cfeature
ds = xr.open_dataset('dataset/space_weather.nc')
tec = ds['TEC'].values
rLat_rad = np.deg2rad(ds['rLat'].values)
rLon_rad = np.deg2rad(ds['rLon'].values)
RL, RLON = np.meshgrid(rLat_rad, rLon_rad, indexing='ij')
b = np.deg2rad(45.0)
cb, sb = np.cos(b), np.sin(b)
xp = np.cos(RL)*np.sin(RLON)
yp = np.cos(RL)*np.cos(RLON)
zp = np.sin(RL)
x = xp
y = cb*yp - sb*zp
z = sb*yp + cb*zp
lat_c = np.rad2deg(np.arcsin(np.clip(z, -1, 1)))
lon_c = np.rad2deg(np.arctan2(x, y))
levels = np.linspace(np.nanmin(tec), np.nanmax(tec), 16)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=0))
ax.set_extent([-115, 115, -15, 65], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor='#f2efe9')
ax.add_feature(cfeature.OCEAN, facecolor='#dbe9f5')
ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='#555')
cf = ax.contourf(lon_c, lat_c, tec, levels=levels, cmap='viridis',
                 transform=ccrs.PlateCarree(), extend='both')
cbar = fig.colorbar(cf, ax=ax, orientation='vertical', pad=0.02)
cbar.set_label('TEC (1E16 e/m$^2$)')
ax.set_title('Total Electron Content (TEC) - Filled Contour', fontsize=13)
fig.savefig('pred_results/TEC_vis.png', dpi=150, bbox_inches='tight')
plt.close(fig)
