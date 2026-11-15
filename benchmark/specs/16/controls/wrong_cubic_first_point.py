# Wrong: cubic fit at the first grid point (the location used by the benchmark's reference script).
import numpy as np, pandas as pd, xarray as xr
ds = xr.open_dataset("dataset/A1B_north_america.nc")
T = ds["air_temperature"]
years = np.arange(1860, 2100)

y = T[:, 0, 0].values.astype(float)
fit = np.polyval(np.polyfit(years - 1860, y, 3), years - 1860)
pd.DataFrame({"year": years, "fitted_temperature": fit}).to_csv("pred_results/polynomial_fit_pred.csv", index=False)
