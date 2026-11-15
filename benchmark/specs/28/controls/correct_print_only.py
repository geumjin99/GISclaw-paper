"""Correct: only prints the MSE (no variable); it is credited through the captured y_test and
predictions.
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
od = pd.read_csv('dataset/od_data.csv'); se = pd.read_csv('dataset/socioeconomic_data.csv')
agg = od.groupby(['Origin', 'Destination'], as_index=False)['FlowVolume'].sum()
df = agg.merge(se.add_prefix('O_'), left_on='Origin', right_on='O_Subregion').merge(se.add_prefix('D_'), left_on='Destination', right_on='D_Subregion')
X = df.select_dtypes('number').drop(columns='FlowVolume'); y = df['FlowVolume']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
gs = GridSearchCV(RandomForestRegressor(random_state=1), {'n_estimators': [100], 'min_samples_leaf': [1, 5]}, cv=4,
                  scoring='neg_mean_squared_error').fit(X_train, y_train)
preds = gs.predict(X_test)
print('Test MSE:', mean_squared_error(y_test, preds))
