"""Correct: the full pipeline inside main(); the MSE is printed, as the instruction asks (no file is
requested).
"""
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
def main():
    od = pd.read_csv('dataset/od_data.csv'); se = pd.read_csv('dataset/socioeconomic_data.csv').set_index('Subregion')
    pairs = od.groupby(['Origin', 'Destination'])['FlowVolume'].sum().reset_index()
    X = pd.concat([se.loc[pairs.Origin].add_suffix('_orig').reset_index(drop=True),
                   se.loc[pairs.Destination].add_suffix('_dest').reset_index(drop=True)], axis=1)
    y = pairs['FlowVolume']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=7)
    gs = GridSearchCV(RandomForestRegressor(random_state=7), {'n_estimators': [100, 300], 'max_depth': [None, 8],
                      'min_samples_leaf': [1, 5]}, cv=5, scoring='neg_mean_squared_error', n_jobs=4).fit(X_tr, y_tr)
    print('best', gs.best_params_)
    print('Test MSE: %.1f' % mean_squared_error(y_te, gs.predict(X_te)))
main()
