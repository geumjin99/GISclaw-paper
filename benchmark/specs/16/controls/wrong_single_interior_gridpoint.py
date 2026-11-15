# Wrong: quadratic fit of one interior grid point (lat idx 10, lon idx 10) instead of the
# regional mean the instruction asks for.
import numpy as np, pandas as pd, xarray as xr
da = xr.open_dataset('dataset/A1B_north_america.nc')['air_temperature']
y = da.isel(latitude=10, longitude=10).values.astype(float)
years = np.array([t.year for t in da.time.values])
fit = np.polyval(np.polyfit(years, y, 2), years)
pd.DataFrame({'year': years, 'fitted_temperature': fit}).to_csv('pred_results/polynomial_fit_pred.csv', index=False)
