"""Correct: medians from the listings' own district label; neighbourhoods without a group are dropped
(138 polygons); different permutation seed; string labels.
"""
import pandas as pd, geopandas as gpd, libpysal, esda, numpy as np, matplotlib.pyplot as plt
nb = gpd.read_file('dataset/berlin-neighbourhoods.geojson')
df = pd.read_csv('dataset/berlin-listings.csv', low_memory=False)
pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326)
joined = gpd.sjoin(pts[['price', 'geometry']], nb, predicate='intersects')
med = joined.groupby('neighbourhood_group')['price'].median()
nb['med_price'] = nb['neighbourhood_group'].map(med)
nb = nb.dropna(subset=['med_price']).reset_index(drop=True)
w = libpysal.weights.Queen.from_dataframe(nb)
w.transform = 'R'
lm = esda.Moran_Local(nb['med_price'], w, permutations=4999, seed=7)
lab = np.array(['ns', 'HH', 'LH', 'LL', 'HL'])
nb['cluster'] = np.where(lm.p_sim < 0.05, lab[lm.q], 'ns')
ax = nb.plot(column='cluster', categorical=True, legend=True, cmap='Set1')
plt.savefig('pred_results/moran_local.png')
