# Correct: unweighted domain mean in degrees Celsius, quadratic fit over the years.
import numpy as np, pandas as pd, xarray as xr
da = xr.open_dataset('dataset/A1B_north_america.nc')['air_temperature']
ts = da.mean(['latitude', 'longitude']) - 273.15
yrs = np.array([t.year for t in da.time.values])
c = np.polyfit(yrs, ts.values, 2)
pd.DataFrame({'year': yrs, 'fitted_temp_C': np.polyval(c, yrs)}).to_csv('pred_results/polynomial_fit_pred.csv', index=False)
