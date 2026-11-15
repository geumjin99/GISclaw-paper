"""Wrong: merges only the origin subregion's attributes."""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
od = pd.read_csv('dataset/od_data.csv'); se = pd.read_csv('dataset/socioeconomic_data.csv')
agg = od.groupby(['Origin', 'Destination'], as_index=False).FlowVolume.sum()
m = agg.merge(se, left_on='Origin', right_on='Subregion')
X = m[['Population', 'GDP', 'EmploymentRate', 'AverageIncome', 'EducationIndex', 'HealthcareAccess']]; y = m.FlowVolume
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
gs = GridSearchCV(RandomForestRegressor(random_state=42), {'n_estimators': [100], 'max_depth': [5, 10]}, cv=5,
                  scoring='neg_mean_squared_error').fit(Xtr, ytr)
mse = mean_squared_error(yte, gs.predict(Xte)); print('MSE', mse)
