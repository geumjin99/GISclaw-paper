# Correct: mean profile over a lat/lon box, 5-2000 m, temperature in degrees Celsius, twin x-axes.
import xarray as xr, matplotlib.pyplot as plt
ds = xr.open_dataset('dataset/atlantic_profiles.nc')
box = ds.sel(lat=slice(-9.9, -3.0), lon=[330.5, 335.5, 340.5, 345.5], depth=slice(5, 2000))
prof = box.mean(dim=['lat', 'lon'], skipna=True)
t_c = prof['theta'] - 273.15
fig, ax1 = plt.subplots(figsize=(7, 8)); ax2 = ax1.twiny()
ax1.plot(t_c.values, prof['depth'].values, 'r-'); ax2.plot(prof['salinity'].values, prof['depth'].values, 'b-')
ax1.invert_yaxis(); ax1.set_xlabel('Potential temperature (C)'); ax2.set_xlabel('Salinity (PSU)'); ax1.set_ylabel('Depth (m)')
plt.savefig('pred_results/ocean_profiles_vis.png', dpi=150, bbox_inches='tight')
