"""Wrong: no Random Forest and no GridSearchCV; plain linear regression on the merged pair table."""
import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
od = pd.read_csv('dataset/od_data.csv'); se = pd.read_csv('dataset/socioeconomic_data.csv')
pairs = od.groupby(['Origin', 'Destination'], as_index=False)['FlowVolume'].sum()
o = se.add_prefix('o_').rename(columns={'o_Subregion': 'Origin'}); d = se.add_prefix('d_').rename(columns={'d_Subregion': 'Destination'})
data = pairs.merge(o, on='Origin').merge(d, on='Destination')
X = data.filter(regex='^(o_|d_)'); y = data['FlowVolume']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1)
lr = LinearRegression().fit(X_tr, y_tr)
test_mse = mean_squared_error(y_te, lr.predict(X_te))
print('test MSE', test_mse)
