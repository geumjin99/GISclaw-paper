# Wrong: fits the first grid point but writes the raw temperatures instead of the fitted values.
import numpy as np, pandas as pd, xarray as xr
da = xr.open_dataset('dataset/A1B_north_america.nc')['air_temperature']
ts = da.isel(latitude=0, longitude=0).values
yrs = np.array([t.year for t in da.time.values])
c = np.polyfit(yrs, ts, 2)
print(c)
pd.DataFrame({'Year': yrs, 'Temperature': ts}).to_csv('pred_results/polynomial_fit_pred.csv', index=False)  # writes data, not the fit
