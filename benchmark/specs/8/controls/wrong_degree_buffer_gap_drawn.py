# Wrong: degree buffers (0.01 deg on EPSG:4326) with the gap drawn on the map; the only error is
# the buffer in geographic coordinates.
import geopandas as gpd, matplotlib.pyplot as plt

ety = gpd.read_file('dataset/etobicoke.geojson')
st = gpd.read_file('dataset/fire_station.geojson')
buf = gpd.GeoDataFrame(geometry=[st.buffer(0.01).union_all()], crs=st.crs)
gaps = gpd.overlay(ety, buf, how='difference')
fig, ax = plt.subplots(figsize=(10, 10))
ety.plot(ax=ax, color='white', edgecolor='black')
gaps.plot(ax=ax, color='red')
st.plot(ax=ax, color='k', markersize=4)
plt.savefig('pred_results/Fire_Service_Analysis.png', dpi=100)
