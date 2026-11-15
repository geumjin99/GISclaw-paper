# Wrong: keeps only stations whose MUN_NAME is Etobicoke, ignoring neighbouring stations that
# also serve Etobicoke's edges.
import geopandas as gpd, matplotlib.pyplot as plt
st = gpd.read_file('dataset/fire_station.geojson').to_crs(32617)
st = st[st.MUN_NAME == 'Etobicoke']
ety = gpd.read_file('dataset/etobicoke.geojson').to_crs(32617)
buf = gpd.GeoDataFrame(geometry=st.buffer(2500), crs=32617)
gaps = gpd.overlay(ety, buf, how='difference')
ax = ety.plot(color='lightgrey')
gaps.plot(ax=ax, color='red')
plt.savefig('pred_results/Fire_Service_Analysis.png')
