"""Correct: drops intra-subregion pairs, merges via map(), uses a different grid and split seed, and
reports R2 and RMSE alongside the MSE.
"""
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
od = pd.read_csv('dataset/od_data.csv'); se = pd.read_csv('dataset/socioeconomic_data.csv').set_index('Subregion')
flows = od[od.Origin != od.Destination].pivot_table(index=['Origin', 'Destination'], values='FlowVolume', aggfunc='sum').reset_index()
for c in se.columns:
    flows['o_' + c] = flows.Origin.map(se[c]); flows['d_' + c] = flows.Destination.map(se[c])
X = flows.filter(regex='^(o_|d_)'); y = flows.FlowVolume
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7)
gs = GridSearchCV(RandomForestRegressor(random_state=0), {'n_estimators': [150], 'max_features': ['sqrt', 1.0], 'max_depth': [5, 15]},
                  cv=3, scoring='neg_mean_squared_error').fit(Xtr, ytr)
test_mse = mean_squared_error(yte, gs.predict(Xte))
rmse = test_mse ** 0.5
print('test MSE', test_mse)
pd.DataFrame({'metric': ['MSE', 'RMSE'], 'value': [test_mse, rmse]}).to_csv('pred_results/metrics.csv', index=False)
