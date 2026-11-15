# Wrong: buffers in geographic coordinates with a degree distance (0.02 deg ~ 2.2 km N-S but only
# ~1.6 km E-W at Toronto's latitude) - distorted service areas.
import geopandas as gpd, matplotlib.pyplot as plt
st = gpd.read_file('dataset/fire_station.geojson')
ety = gpd.read_file('dataset/etobicoke.geojson')
buf = gpd.GeoDataFrame(geometry=[st.buffer(0.025).union_all()], crs=4326)
gaps = gpd.overlay(ety, buf, how='difference')
ax = ety.plot(color='lightgrey')
gaps.plot(ax=ax, color='red')
plt.savefig('pred_results/Fire_Service_Analysis.png')
