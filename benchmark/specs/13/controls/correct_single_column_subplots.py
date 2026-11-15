# Correct: single column read with netCDF4 (340.5E, 6.5S, depth <= 1500 m), temperature and
# salinity in two side-by-side panels sharing the depth axis.
import netCDF4, numpy as np, matplotlib.pyplot as plt
nc = netCDF4.Dataset('dataset/atlantic_profiles.nc')
lat = nc['lat'][:]; lon = nc['lon'][:]; dep = nc['depth'][:]
i = int(np.argmin(abs(lat + 6.5))); j = int(np.argmin(abs(lon - 340.5)))
keep = dep <= 1500
temp = nc['theta'][keep, i, j]; sal = nc['salinity'][keep, i, j]
fig, (a, b) = plt.subplots(1, 2, sharey=True, figsize=(8, 6))
a.plot(temp, dep[keep]); a.set_xlabel('theta (K)'); a.invert_yaxis(); a.set_ylabel('depth (m)')
b.plot(sal, dep[keep], color='r'); b.set_xlabel('salinity')
plt.savefig('pred_results/ocean_profiles_vis.png')
