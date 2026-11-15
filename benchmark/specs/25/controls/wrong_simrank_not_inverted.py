"""Wrong: joins SIMRANK itself (1 = most at risk) instead of 432 - SIMRANK, and clusters only two risks."""
import numpy as np, geopandas as gpd
from sklearn.cluster import KMeans
h = gpd.read_file('dataset/hkg_constituency.shp'); t = gpd.read_file('dataset/target_risk.shp')
sets = {'trans_risk': ['PopDensity', 'SpInterIdx'], 'susc_risk': ['PopDensity', 'Seniors60t', 'spending_c', 'keyfacts_p'],
        'insr_risk': ['Seniors60t', 'SUM_Beds'], 'expo_risk': ['RelativeCa']}
for f, cols in sets.items():
    X = h[cols].values.astype(float); T = t[cols].values.astype(float)
    mu, sd = X.mean(0), X.std(0)
    d = (((X - T) / sd) ** 2).sum(1)
    h[f] = np.argsort(np.argsort(d)) + 1
h['cluster'] = KMeans(3, n_init=10, random_state=0).fit_predict(h[['trans_risk', 'susc_risk']])
h.plot(column='cluster', categorical=True).figure.savefig('pred_results/covid_risk.png')
