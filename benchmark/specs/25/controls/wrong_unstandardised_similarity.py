"""Wrong: similarity on raw (unstandardised) attribute differences, so PopDensity dominates."""
import numpy as np, geopandas as gpd
from sklearn.cluster import KMeans
h = gpd.read_file('dataset/hkg_constituency.shp'); t = gpd.read_file('dataset/target_risk.shp')
sets = {'trans_risk': ['PopDensity', 'SpInterIdx'], 'susc_risk': ['PopDensity', 'Seniors60t', 'spending_c', 'keyfacts_p'],
        'insr_risk': ['Seniors60t', 'SUM_Beds'], 'expo_risk': ['RelativeCa']}
for f, cols in sets.items():
    d = np.sqrt(((h[cols].values - t[cols].values) ** 2).sum(1))
    h[f] = 432 - (np.argsort(np.argsort(d)) + 1)
seed = h.index[h.SEEDS == 1]
X = h[list(sets)].values.astype(float)
h['cluster'] = KMeans(3, init=X[seed], n_init=1).fit_predict(X)
h.plot(column='cluster', categorical=True, legend=True).figure.savefig('pred_results/covid_risk.png')
