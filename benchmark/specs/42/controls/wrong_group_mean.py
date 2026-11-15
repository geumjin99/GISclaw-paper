"""Wrong: group MEAN price (named median) instead of the median the instruction asks for."""
import pandas as pd, geopandas as gpd, libpysal as lps, esda, numpy as np, matplotlib.pyplot as plt
gdf = gpd.read_file('dataset/berlin-neighbourhoods.geojson')
bl = pd.read_csv('dataset/berlin-listings.csv', low_memory=False)
bl_gdf = gpd.GeoDataFrame(bl, geometry=gpd.points_from_xy(bl.longitude, bl.latitude), crs=4326)
sj = gpd.sjoin(gdf, bl_gdf, how='inner', predicate='intersects')
m = sj['price'].groupby(sj['neighbourhood_group']).mean()
gdf = gdf.join(m, on='neighbourhood_group').rename(columns={'price': 'median_pri'})
gdf['median_pri'] = gdf['median_pri'].fillna(gdf['median_pri'].mean())
wq = lps.weights.Queen.from_dataframe(gdf); wq.transform = 'r'
li = esda.moran.Moran_Local(gdf['median_pri'], wq, seed=1)
gdf['spots'] = np.where(li.p_sim < 0.05, li.q, 0)
gdf.plot(column='spots', categorical=True, legend=True)
plt.savefig('pred_results/moran_local.png')
