# Correct: area-weighted mean, quadratic fit, values reported to 0.1 K.
import numpy as np, pandas as pd, xarray as xr
ds = xr.open_dataset("dataset/A1B_north_america.nc")
T = ds["air_temperature"]
years = np.arange(1860, 2100)

w = np.cos(np.deg2rad(T.latitude))
y = T.weighted(w).mean(("latitude", "longitude")).values
fit = np.polyval(np.polyfit(years - 1860, y, 2), years - 1860)
pd.DataFrame({"year": years, "fitted_temperature_K": np.round(fit, 1)}).to_csv("pred_results/polynomial_fit_pred.csv", index=False)
