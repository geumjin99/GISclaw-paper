# Correct: explicit loop counting distinct runs of >= 5 warm years; drawn with imshow(origin='lower').
import numpy as np, netCDF4, matplotlib.pyplot as plt
def count_runs(series, thr=280.0, length=5):
    n = run = 0
    for v in series:
        run = run + 1 if v > thr else 0
        if run == length:
            n += 1
    return n
with netCDF4.Dataset('dataset/E1_north_america.nc') as nc:
    T = nc['air_temperature'][:].filled(np.nan)
    lat, lon = nc['latitude'][:], nc['longitude'][:]
occ = np.array([[count_runs(T[:, i, j]) for j in range(T.shape[2])] for i in range(T.shape[1])])
plt.figure(figsize=(9, 6))
plt.imshow(occ, origin='lower', extent=(lon.min(), lon.max(), lat.min(), lat.max()), cmap='YlOrRd')
plt.colorbar(label='Number of >=5-year runs above 280 K')
plt.title('Warm spells 1860-2099')
plt.savefig('pred_results/temperature_statistic_vis.png')
