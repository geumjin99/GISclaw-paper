# Wrong: area weights np.cos(latitude) with latitude in degrees (not radians) - oscillating, partly negative weights.
import numpy as np, pandas as pd, xarray as xr
ds = xr.open_dataset("dataset/A1B_north_america.nc")
T = ds["air_temperature"]
years = np.arange(1860, 2100)

w = np.cos(T.latitude.values)[None, :, None] * np.ones(T.shape)
y = (T.values * w).sum((1, 2)) / w.sum((1, 2))
fit = np.polyval(np.polyfit(years, y, 2), years)
pd.DataFrame({"year": years, "fitted_temperature": fit}).to_csv("pred_results/polynomial_fit_pred.csv", index=False)
