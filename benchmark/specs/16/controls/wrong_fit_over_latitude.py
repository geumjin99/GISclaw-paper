# Wrong: fits the polynomial along latitude (time-mean profile) instead of over the years.
import numpy as np, pandas as pd, xarray as xr
da = xr.open_dataset('dataset/A1B_north_america.nc')['air_temperature']
prof = da.mean(['time', 'longitude'])            # fits along latitude instead of time
c = np.polyfit(prof.latitude.values, prof.values, 2)
pd.DataFrame({'Year': prof.latitude.values, 'Fitted_Temperature': np.polyval(c, prof.latitude.values)}).to_csv('pred_results/polynomial_fit_pred.csv', index=False)
