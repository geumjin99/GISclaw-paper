# Wrong: fits the yearly domain MAXIMUM temperature instead of a temperature series of the region.
import numpy as np, pandas as pd, xarray as xr
ds = xr.open_dataset("dataset/A1B_north_america.nc")
T = ds["air_temperature"]
years = np.arange(1860, 2100)

y = T.max(("latitude", "longitude")).values
fit = np.polyval(np.polyfit(years, y, 2), years)
pd.DataFrame({"year": years, "fitted_temperature": fit}).to_csv("pred_results/polynomial_fit_pred.csv", index=False)
