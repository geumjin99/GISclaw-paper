# Wrong: district reprojected to UTM but stations buffered by 0.0225 degrees in WGS84, then reprojected.
import geopandas as gpd, matplotlib.pyplot as plt
st = gpd.read_file('dataset/fire_station.geojson')
et = gpd.read_file('dataset/etobicoke.geojson').to_crs(32617)
buf = gpd.GeoDataFrame(geometry=st.buffer(0.0225), crs=4326).to_crs(32617).dissolve()
gaps = gpd.overlay(et, buf, how='difference')
ax = et.plot(color='lightgrey', figsize=(10, 10))
gaps.plot(ax=ax, color='red')
plt.savefig('pred_results/Fire_Service_Analysis.png')
