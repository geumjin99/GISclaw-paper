# Correct: one column, temperature in degC and salinity on the same x-axis (salinity spans
# ~4-5 % of the axis, still legible), depth inverted.
import xarray as xr, matplotlib.pyplot as plt
ds = xr.open_dataset('dataset/atlantic_profiles.nc')
c = ds.sel(lon=330.5, lat=-9.834, method='nearest').sel(depth=slice(0, 1000))
plt.figure(figsize=(6, 8))
plt.plot(c.theta.values - 273.15, c.depth.values, label='Temperature (degC)')
plt.plot(c.salinity.values, c.depth.values, label='Salinity (PSU)')
plt.gca().invert_yaxis(); plt.ylabel('Depth (m)'); plt.legend()
plt.savefig('pred_results/ocean_profiles_vis.png')
