"""Reference solution for task 16: quadratic trend of North American annual mean temperature
(A1B scenario), 1860-2099.

The fitted series is the area-weighted (cos latitude) mean
over the whole North America grid, the regional mean of "climate change trends in North
America". The benchmark's reference script fits the first grid point (15N, 135W) instead; the
checks accept the area-weighted or the unweighted domain mean.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr

os.makedirs("pred_results", exist_ok=True)
ds = xr.open_dataset("dataset/A1B_north_america.nc")
temp = ds["air_temperature"]                       # (time, latitude, longitude), K
weights = np.cos(np.deg2rad(temp["latitude"]))
series = temp.weighted(weights).mean(("latitude", "longitude"))   # area-weighted mean, K

years = np.array([t.year for t in pd.to_datetime([str(v)[:10] for v in temp["time"].values])])
y = series.values.astype("float64")
coeffs = np.polyfit(years - years[0], y, 2)       # centred x for numerical conditioning
fitted = np.polyval(coeffs, years - years[0])
print("quadratic coefficients (K/yr^2, K/yr, K):", coeffs)
print("fitted 1860 %.3f K, 2099 %.3f K; residual sd %.3f K" % (fitted[0], fitted[-1], np.std(y - fitted)))

result_df = pd.DataFrame({"Year": years, "Observed_Temperature": np.round(y, 4),
                          "Fitted_Temperature": np.round(fitted, 4)})
result_df.to_csv("pred_results/polynomial_fit_pred.csv", index=False)
