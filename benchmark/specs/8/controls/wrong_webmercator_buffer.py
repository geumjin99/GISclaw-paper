# Wrong: "metric" buffer of 2500 m done in Web Mercator (EPSG:3857); at 43.7 N the true radius is ~1.8 km.
import geopandas as gpd, matplotlib.pyplot as plt
st = gpd.read_file('dataset/fire_station.geojson').to_crs(3857)
et = gpd.read_file('dataset/etobicoke.geojson').to_crs(3857)
buf = gpd.GeoDataFrame(geometry=[st.buffer(2500).union_all()], crs=3857)
gaps = gpd.overlay(et, buf, how='difference')
ax = et.plot(color='lightgrey', figsize=(10, 10))
gaps.plot(ax=ax, color='red')
st.plot(ax=ax, color='k', markersize=5)
ax.set_title('Areas more than 2.5 km from a fire station')
plt.savefig('pred_results/Fire_Service_Analysis.png')
