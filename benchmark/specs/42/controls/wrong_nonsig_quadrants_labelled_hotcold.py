"""Wrong: every HH / LL neighbourhood is labelled 'Hot spot' / 'Cold spot' regardless of significance;
the p_sim column sits beside it but is unused.
"""
import pandas as pd, geopandas as gpd, libpysal as lps, esda, numpy as np, matplotlib.pyplot as plt
gdf = gpd.read_file('dataset/berlin-neighbourhoods.geojson')
bl = pd.read_csv('dataset/berlin-listings.csv', low_memory=False)
bl_gdf = gpd.GeoDataFrame(bl[['price']], geometry=gpd.points_from_xy(bl.longitude, bl.latitude), crs=4326)
sj = gpd.sjoin(bl_gdf, gdf, predicate='within')
gdf['median_price'] = gdf['neighbourhood_group'].map(sj.groupby('neighbourhood_group')['price'].median())
gdf['median_price'] = gdf['median_price'].fillna(gdf['median_price'].mean())
wq = lps.weights.Queen.from_dataframe(gdf); wq.transform = 'r'
li = esda.moran.Moran_Local(gdf['median_price'], wq, permutations=999, seed=12345)
lab = {1: 'Hot spot', 2: 'Outlier', 3: 'Cold spot', 4: 'Outlier'}
gdf['cluster'] = [lab[int(x)] for x in li.q]
gdf['p_sim'] = li.p_sim
gdf.plot(column='cluster', categorical=True, legend=True)
plt.savefig('pred_results/moran_local.png')
