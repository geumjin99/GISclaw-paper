# Correct: quadratic fit of the domain mean,
# written from an iris cube -> DataFrame, so the CSV carries iris auxiliary coordinates first
# (forecast_period, forecast_reference_time = 1859-09-01 on every row, height) before 'year'.
import iris, numpy as np, pandas as pd
cube = iris.load('dataset/A1B_north_america.nc')[0]
mean = cube.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)
t = cube.coord('time')
years = np.array([d.year for d in t.units.num2date(t.points)])
frt = cube.coord('forecast_reference_time')
frt_s = str(frt.units.num2date(frt.points[0]))
coef = np.polyfit(np.arange(len(years)), mean.data, 2)
df = pd.DataFrame({'forecast_period': [f'{i} days 18:00:00' for i in range(len(years))],
                   'forecast_reference_time': frt_s, 'height': 1.5,
                   'temperature': mean.data, 'year': years,
                   'fitted_temperature': np.polyval(coef, np.arange(len(years)))})
df.to_csv('pred_results/polynomial_fit_pred.csv', index=False)
