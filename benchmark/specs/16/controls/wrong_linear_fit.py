# Wrong: linear (degree-1) fit instead of the requested quadratic.
import numpy as np, pandas as pd, xarray as xr
da = xr.open_dataset('dataset/A1B_north_america.nc')['air_temperature']
ts = da.mean(['latitude', 'longitude']).values
yrs = np.array([t.year for t in da.time.values])
c = np.polyfit(yrs, ts, 1)                       # degree 1 instead of 2
pd.DataFrame({'Year': yrs, 'Fitted_Temperature': np.polyval(c, yrs)}).to_csv('pred_results/polynomial_fit_pred.csv', index=False)
