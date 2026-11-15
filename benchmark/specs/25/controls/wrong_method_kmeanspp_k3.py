"""Wrong: K-means with k-means++ (k = 3, no seed field) on unstandardised rank scores, whereas the
task prescribes K-means initialised from the SEEDS field. Similarity via scipy cdist on z-scores
(pooled with the target), ranks via argsort.
"""
import numpy as np, geopandas as gpd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
h = gpd.read_file('dataset/hkg_constituency.shp'); t = gpd.read_file('dataset/target_risk.shp')
sets = {'transmission': ['PopDensity', 'SpInterIdx'], 'susceptibility': ['PopDensity', 'Seniors60t', 'spending_c', 'keyfacts_p'],
        'scarcity': ['Seniors60t', 'SUM_Beds'], 'exposure': ['RelativeCa']}
for name, cols in sets.items():
    A = np.vstack([h[cols].values, t[cols].values]).astype(float)
    A = (A - A.mean(0)) / A.std(0)
    d = cdist(A[:-1], A[-1:]).ravel()
    simrank = np.empty(len(d), int); simrank[np.argsort(d, kind='stable')] = np.arange(1, len(d) + 1)
    h[name] = 432 - simrank
h['profile'] = KMeans(3, n_init=20, random_state=1).fit_predict(h[list(sets)].values)
ax = h.plot(column='profile', categorical=True, legend=True)
ax.figure.savefig('pred_results/covid_risk_profiles.png')
h.to_file('pred_results/risk_profiles.gpkg')
