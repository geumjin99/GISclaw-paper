# Wrong: linear instead of quadratic fit, with an extra extrapolated 2100 row (the extra row is ignored).
import numpy as np, pandas as pd, xarray as xr
da = xr.open_dataset('dataset/A1B_north_america.nc')['air_temperature']
ts = da.mean(['latitude', 'longitude'])
yrs = np.array([t.year for t in da.time.values])
c = np.polyfit(yrs, ts.values, 1)
yy = np.append(yrs, 2100)
pd.DataFrame({'year': yy, 'fitted_temp_K': np.polyval(c, yy)}).to_csv('pred_results/polynomial_fit_pred.csv', index=False)
