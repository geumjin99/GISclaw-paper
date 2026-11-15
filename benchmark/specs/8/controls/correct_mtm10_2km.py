# Correct: NAD83 / MTM zone 10 (Toronto's local CRS), 2 km buffers, gap via shapely difference.
import geopandas as gpd, matplotlib.pyplot as plt
st = gpd.read_file('dataset/fire_station.geojson').to_crs(2019)
ety = gpd.read_file('dataset/etobicoke.geojson').to_crs(2019)
covered = st.buffer(2000).union_all()
gaps = gpd.GeoDataFrame(geometry=[ety.union_all().difference(covered)], crs=2019).explode(index_parts=False)
ax = ety.plot(color='lightgrey', edgecolor='k')
gaps.plot(ax=ax, color='red')
st.plot(ax=ax, color='blue', markersize=5)
plt.savefig('pred_results/Fire_Service_Analysis.png')
