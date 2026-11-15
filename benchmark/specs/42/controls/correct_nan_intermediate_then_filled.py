"""Correct: an intermediate table with the two group-less medians still NaN stays in the namespace;
the analysis runs on a filled copy (gap filled with the median of the group medians). The NaN
intermediate does not decide the missing-data check.
"""
import pandas as pd, geopandas as gpd, libpysal, esda, numpy as np, matplotlib.pyplot as plt
nb = gpd.read_file('dataset/berlin-neighbourhoods.geojson')
df = pd.read_csv('dataset/berlin-listings.csv', low_memory=False)
pts = gpd.GeoDataFrame(df[['price']], geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326)
sj = gpd.sjoin(pts, nb, predicate='within')
med = sj.groupby('neighbourhood_group')['price'].median()
a_raw = nb.copy()
a_raw['median_price'] = a_raw['neighbourhood_group'].map(med)          # 2 NaN left here
print('missing:', int(a_raw['median_price'].isna().sum()))
filled = a_raw.copy()
filled['median_price'] = filled['median_price'].fillna(filled['median_price'].median())
w = libpysal.weights.Queen.from_dataframe(filled, use_index=False); w.transform = 'r'
li = esda.moran.Moran_Local(filled['median_price'].values, w, permutations=999, seed=3)
lab = np.array(['not significant', 'HH hot spot', 'LH outlier', 'LL cold spot', 'HL outlier'])
filled['lisa'] = np.where(li.p_sim < 0.05, lab[li.q], 'not significant')
filled.plot(column='lisa', categorical=True, legend=True)
plt.savefig('pred_results/moran_local.png')
