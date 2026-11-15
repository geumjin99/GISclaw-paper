# Wrong: only the 22 stations inside the district, 1.5 km buffers. The gap is close to the
# all-station 1.5 km gap (IoU ~0.94) but keeps ~4.6 % of areas within reach of stations outside
# the district, which also serve its edges.
import geopandas as gpd, matplotlib.pyplot as plt
st = gpd.read_file('dataset/fire_station.geojson').to_crs(32617)
ety = gpd.read_file('dataset/etobicoke.geojson').to_crs(32617)
st = st[st.within(ety.union_all())]
gaps = gpd.overlay(ety, gpd.GeoDataFrame(geometry=[st.buffer(1500).union_all()], crs=32617), how='difference')
ax = ety.plot(color='lightgrey', edgecolor='k', figsize=(9, 9))
gaps.plot(ax=ax, color='red')
plt.savefig('pred_results/Fire_Service_Analysis.png')
