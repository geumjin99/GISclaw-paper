# Wrong: cubic (degree-3) polynomial instead of the requested quadratic, area-weighted mean.
import numpy as np, pandas as pd, xarray as xr
ds = xr.open_dataset("dataset/A1B_north_america.nc")
T = ds["air_temperature"]
years = np.arange(1860, 2100)

w = np.cos(np.deg2rad(T.latitude))
y = T.weighted(w).mean(("latitude", "longitude")).values
fit = np.polyval(np.polyfit(years - 1860, y, 3), years - 1860)
pd.DataFrame({"year": years, "fitted_temperature": fit}).to_csv("pred_results/polynomial_fit_pred.csv", index=False)
