# Correct analysis (regional-mean T and S profiles of one box, drawn against depth); only the
# presentation is wrong: both are drawn on ONE axis (temperature 275-300 K, salinity 34-36), so
# the salinity profile is a flat line occupying < 1 % of the axis, and the axis labels are
# swapped. Fails only the kind="figure" legibility check.
import xarray as xr, matplotlib.pyplot as plt
ds = xr.open_dataset('dataset/atlantic_profiles.nc')
sub = ds.sel(depth=slice(5, 4478))
t = sub.theta.mean(dim=('lat', 'lon'), skipna=True)
s = sub.salinity.mean(dim=('lat', 'lon'), skipna=True)
fig = plt.figure(figsize=(10, 6))
plt.plot(t.values, sub.depth.values, label='Temperature')
plt.plot(s.values, sub.depth.values, label='Salinity')
plt.gca().invert_yaxis()
plt.xlabel('Depth'); plt.ylabel('Value'); plt.legend()
plt.savefig('pred_results/ocean_profiles_vis.png', dpi=150, bbox_inches='tight')
