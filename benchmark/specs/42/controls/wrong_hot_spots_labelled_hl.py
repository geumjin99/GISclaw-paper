"""Wrong: correct clusters, but the hot-spot set is labelled as a high-low outlier ('HL') and the
cold-spot set as 'LH'; the quadrant labels are swapped.
"""
import pandas as pd, geopandas as gpd, libpysal, esda, numpy as np, matplotlib.pyplot as plt
nb = gpd.read_file('dataset/berlin-neighbourhoods.geojson')
df = pd.read_csv('dataset/berlin-listings.csv', low_memory=False)
pts = gpd.GeoDataFrame(df[['price']], geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326)
sj = gpd.sjoin(pts, nb, predicate='within')
nb['med'] = nb['neighbourhood_group'].map(sj.groupby('neighbourhood_group')['price'].median())
nb['med'] = nb['med'].fillna(nb['med'].mean())
w = libpysal.weights.Queen.from_dataframe(nb, use_index=False); w.transform = 'r'
li = esda.moran.Moran_Local(nb['med'].values, w, permutations=999, seed=12345)
lab = np.array(['ns', 'HL', 'HH', 'LH', 'LL'])      # q=1 (HH) -> 'HL', q=3 (LL) -> 'LH'
nb['cluster'] = np.where(li.p_sim < 0.05, lab[li.q], 'ns')
nb.plot(column='cluster', categorical=True, legend=True)
plt.savefig('pred_results/moran_local.png')
