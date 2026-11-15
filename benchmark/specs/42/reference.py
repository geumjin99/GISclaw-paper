"""Reference solution for task 42: Local Moran's I of Airbnb prices in Berlin.

Listings -> points (WGS84) -> spatial join with the 140 neighbourhood polygons -> MEDIAN price
per neighbourhood_group (12 districts) -> assigned to every neighbourhood of that group ->
the two neighbourhoods with no group (missing data) get the mean of the group medians ->
Queen contiguity (row-standardised) on the 140 neighbourhoods -> Local Moran's I (999
permutations, fixed seed) -> significant (p < 0.05) HH / LH / LL / HL clusters mapped.

Adapted from the reference script of GeoAnalystBench (Zhang et al. 2025, Apache-2.0).
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import libpysal as lps
import esda
import matplotlib.pyplot as plt
import matplotlib.colors as colors

nbhd = gpd.read_file('dataset/berlin-neighbourhoods.geojson')
listings = pd.read_csv('dataset/berlin-listings.csv', low_memory=False)
listings_gdf = gpd.GeoDataFrame(listings[['id', 'price']].copy(),
                                geometry=gpd.points_from_xy(listings.longitude, listings.latitude),
                                crs='EPSG:4326')
listings_gdf['price'] = listings_gdf['price'].astype(float)

# spatial join: listings -> neighbourhood polygons (both EPSG:4326)
sj = gpd.sjoin(listings_gdf, nbhd, how='inner', predicate='within')
median_price = sj.groupby('neighbourhood_group')['price'].median().rename('median_pri')

nbhd = nbhd.join(median_price, on='neighbourhood_group')
n_missing = int(nbhd['median_pri'].isna().sum())       # 2 polygons have no neighbourhood_group
nbhd['median_pri'] = nbhd['median_pri'].fillna(nbhd['median_pri'].mean())
print('missing medians filled:', n_missing)

# spatial weights + Local Moran's I
wq = lps.weights.Queen.from_dataframe(nbhd, use_index=False)
wq.transform = 'r'
y = nbhd['median_pri'].values
li = esda.moran.Moran_Local(y, wq, permutations=999, seed=12345)
sig = li.p_sim < 0.05
nbhd['lisa_q'] = li.q
nbhd['lisa_p'] = li.p_sim
nbhd['lisa_cluster'] = np.where(sig, li.q, 0)          # 0 ns, 1 HH (hot), 2 LH, 3 LL (cold), 4 HL
labels = {0: '0 not significant', 1: '1 hot spot (HH)', 2: '2 doughnut (LH)', 3: '3 cold spot (LL)', 4: '4 diamond (HL)'}
nbhd['lisa_label'] = nbhd['lisa_cluster'].map(labels)
print(nbhd['lisa_label'].value_counts())

present = sorted(nbhd['lisa_cluster'].unique())
palette = {0: 'lightgrey', 1: 'red', 2: 'lightblue', 3: 'blue', 4: 'pink'}
cmap = colors.ListedColormap([palette[k] for k in present])
fig, ax = plt.subplots(1, figsize=(9, 9))
nbhd.plot(column='lisa_label', categorical=True, cmap=cmap, linewidth=0.2, ax=ax,
          edgecolor='white', legend=True)
ax.set_axis_off()
ax.set_title("Local Moran's I of median Airbnb price (significant clusters, p < 0.05)")
plt.savefig('pred_results/moran_local.png', dpi=120, bbox_inches='tight')
