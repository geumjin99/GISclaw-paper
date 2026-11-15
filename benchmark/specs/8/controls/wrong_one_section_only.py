# Wrong: takes only the first feature of the district layer (Central Section) as "Etobicoke".
import geopandas as gpd, matplotlib.pyplot as plt
st = gpd.read_file('dataset/fire_station.geojson').to_crs(32617)
et = gpd.read_file('dataset/etobicoke.geojson').to_crs(32617).iloc[[0]]
buf = gpd.GeoDataFrame(geometry=[st.buffer(2500).union_all()], crs=32617)
gaps = gpd.overlay(et, buf, how='difference')
ax = et.plot(color='lightgrey', figsize=(10, 10))
gaps.plot(ax=ax, color='red')
plt.savefig('pred_results/Fire_Service_Analysis.png')
