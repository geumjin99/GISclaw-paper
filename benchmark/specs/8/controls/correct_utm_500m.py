# Correct: short service radius - UTM 17N, 500 m buffers around all stations, overlay difference,
# map the gap (the task gives no buffer distance; the accepted range starts at 400 m).
import geopandas as gpd, matplotlib.pyplot as plt
st = gpd.read_file('dataset/fire_station.geojson').to_crs(32617)
ety = gpd.read_file('dataset/etobicoke.geojson').to_crs(32617)
buf = gpd.GeoDataFrame(geometry=[st.buffer(500).union_all()], crs=32617)
gaps = gpd.overlay(ety, buf, how='difference')
ax = ety.plot(color='lightgrey', edgecolor='k', figsize=(9, 9))
gaps.plot(ax=ax, color='red', alpha=0.6)
st.plot(ax=ax, color='blue', markersize=5)
ax.set_title('Etobicoke areas more than 500 m from a fire station')
plt.savefig('pred_results/Fire_Service_Analysis.png')
