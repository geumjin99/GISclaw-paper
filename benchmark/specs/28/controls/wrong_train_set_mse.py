"""Wrong: correct preparation, but the tuned model is evaluated on its own training data."""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
od = pd.read_csv('dataset/od_data.csv'); se = pd.read_csv('dataset/socioeconomic_data.csv')
agg = od.groupby(['Origin', 'Destination'], as_index=False).FlowVolume.sum()
m = agg.merge(se, left_on='Origin', right_on='Subregion').merge(se, left_on='Destination', right_on='Subregion', suffixes=('_o', '_d'))
X = m.select_dtypes('number').drop(columns='FlowVolume'); y = m.FlowVolume
gs = GridSearchCV(RandomForestRegressor(random_state=42), {'n_estimators': [100], 'max_depth': [10, None]}, cv=5,
                  scoring='neg_mean_squared_error').fit(X, y)
mse = mean_squared_error(y, gs.best_estimator_.predict(X)); print('MSE', mse)
