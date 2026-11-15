# Wrong: 400 m buffers are built, but the layer mapped as "gap" is the district outline itself.
import geopandas as gpd, matplotlib.pyplot as plt
st = gpd.read_file('dataset/fire_station.geojson').to_crs(32617)
ety = gpd.read_file('dataset/etobicoke.geojson').to_crs(32617)
buf = st.buffer(400)
gaps = ety.dissolve()
ax = gaps.plot(color='red', edgecolor='k', figsize=(9, 9))
plt.savefig('pred_results/Fire_Service_Analysis.png')
