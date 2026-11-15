"""Correct: the key columns are renamed orig / dest and the metrics are written as a one-row wide CSV."""
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
od = pd.read_csv('dataset/od_data.csv').rename(columns={'Origin': 'orig', 'Destination': 'dest', 'FlowVolume': 'flow'})
se = pd.read_csv('dataset/socioeconomic_data.csv')
flows = od.groupby(['orig', 'dest'], as_index=False)['flow'].sum()
tab = flows.merge(se.add_suffix('_o'), left_on='orig', right_on='Subregion_o').merge(se.add_suffix('_d'), left_on='dest', right_on='Subregion_d')
tab = tab.drop(columns=['Subregion_o', 'Subregion_d'])
feat = [c for c in tab.columns if c.endswith(('_o', '_d'))]
X_train, X_test, y_train, y_test = train_test_split(tab[feat], tab['flow'], test_size=0.3, random_state=3)
gs = GridSearchCV(RandomForestRegressor(random_state=3), {'n_estimators': [150], 'max_depth': [5, 15, None]}, cv=4,
                  scoring='neg_mean_squared_error').fit(X_train, y_train)
pred = gs.predict(X_test)
pd.DataFrame([{'train_mse': mean_squared_error(y_train, gs.predict(X_train)), 'test_mse': mean_squared_error(y_test, pred),
               'test_r2': r2_score(y_test, pred)}]).to_csv('pred_results/metrics.csv', index=False)
tab.to_csv('pred_results/interaction_table.csv', index=False)
