# Correct: quadratic fit of the unweighted domain mean, plus one extrapolated row for 2100
# (scored on 1860-2099).
import numpy as np, pandas as pd, xarray as xr
da = xr.open_dataset('dataset/A1B_north_america.nc')['air_temperature']
ts = da.mean(['latitude', 'longitude'])
yrs = np.array([t.year for t in da.time.values])
c = np.polyfit(yrs, ts.values, 2)
yy = np.append(yrs, 2100)
pd.DataFrame({'year': yy, 'fitted_temp_K': np.polyval(c, yy)}).to_csv('pred_results/polynomial_fit_pred.csv', index=False)
