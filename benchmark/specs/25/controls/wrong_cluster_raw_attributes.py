"""Wrong: skips the similarity / risk step and clusters the raw attributes directly."""
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
h = gpd.read_file('dataset/hkg_constituency.shp')
cols = ['PopDensity', 'SpInterIdx', 'Seniors60t', 'spending_c', 'keyfacts_p', 'SUM_Beds', 'RelativeCa']
X = StandardScaler().fit_transform(h[cols])
h['risk_cluster'] = KMeans(3, n_init=10, random_state=0).fit_predict(X)
h.plot(column='risk_cluster', categorical=True, legend=True).figure.savefig('pred_results/covid_risk.png')
