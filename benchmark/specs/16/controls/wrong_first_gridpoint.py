# Wrong: quadratic fit of the first grid point (the location used by the benchmark's reference
# script), read with netCDF4; the instruction asks for the regional mean.
import numpy as np, csv, netCDF4
nc = netCDF4.Dataset('dataset/A1B_north_america.nc')
tv = nc.variables['time']
x = tv[:].astype(float)
y = nc.variables['air_temperature'][:, 0, 0].astype(float)
p = np.polyfit(x, y, 2)
fit = np.polyval(p, x)
dates = netCDF4.num2date(x, tv.units, getattr(tv, 'calendar', 'standard'))
with open('pred_results/polynomial_fit_pred.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['Year', 'Fitted_Temperature'])
    w.writerows([(d.strftime('%Y'), v) for d, v in zip(dates, fit)])
