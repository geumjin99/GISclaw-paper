"""Wrong: skips the aggregation and models the 5000 raw records (duplicated pairs) directly."""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
od = pd.read_csv('dataset/od_data.csv'); se = pd.read_csv('dataset/socioeconomic_data.csv')
m = od.merge(se, left_on='Origin', right_on='Subregion').merge(se, left_on='Destination', right_on='Subregion', suffixes=('_o', '_d'))
X = m.select_dtypes('number').drop(columns='FlowVolume'); y = m.FlowVolume
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
gs = GridSearchCV(RandomForestRegressor(random_state=42), {'n_estimators': [100], 'max_depth': [None, 10]}, cv=3,
                  scoring='neg_mean_squared_error').fit(Xtr, ytr)
mse = mean_squared_error(yte, gs.predict(Xte)); print('MSE', mse)
