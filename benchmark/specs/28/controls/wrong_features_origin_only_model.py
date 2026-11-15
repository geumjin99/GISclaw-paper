"""Wrong: the merged table carries origin and destination attributes, but the model's feature list
takes only the origin side (_x).
"""
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
od = pd.read_csv('dataset/od_data.csv'); se = pd.read_csv('dataset/socioeconomic_data.csv')
agg = od.groupby(['Origin', 'Destination'], as_index=False)['FlowVolume'].sum()
m = agg.merge(se, left_on='Origin', right_on='Subregion').merge(se, left_on='Destination', right_on='Subregion')
feats = [c for c in m.columns if c.endswith('_x') and c != 'Subregion_x']
X_tr, X_te, y_tr, y_te = train_test_split(m[feats], m['FlowVolume'], test_size=0.2, random_state=0)
gs = GridSearchCV(RandomForestRegressor(random_state=0), {'n_estimators': [100, 200], 'max_depth': [None, 10]}, cv=3,
                  scoring='neg_mean_squared_error').fit(X_tr, y_tr)
mse = mean_squared_error(y_te, gs.best_estimator_.predict(X_te))
print('Test MSE', mse)
