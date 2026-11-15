# Wrong: difference taken the wrong way round (station buffers minus Etobicoke) and mapped as "gaps".
import geopandas as gpd, matplotlib.pyplot as plt
st = gpd.read_file('dataset/fire_station.geojson').to_crs(32617)
ety = gpd.read_file('dataset/etobicoke.geojson').to_crs(32617)
buf = gpd.GeoDataFrame(geometry=[st.buffer(2500).union_all()], crs=32617)
gaps = gpd.overlay(buf, ety, how='difference')
ax = ety.plot(color='lightgrey')
gaps.plot(ax=ax, color='red')
plt.savefig('pred_results/Fire_Service_Analysis.png')
