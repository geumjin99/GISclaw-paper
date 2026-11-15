# Correct: gap computed in UTM 17N with 2 km buffers, but drawn reprojected inline to WGS84 and as
# an outline (boundary) over a filled district.
import geopandas as gpd, matplotlib.pyplot as plt

ety = gpd.read_file('dataset/etobicoke.geojson').to_crs(32617)
st = gpd.read_file('dataset/fire_station.geojson').to_crs(32617)
cov = st.buffer(2000).union_all()
gaps = gpd.GeoDataFrame(geometry=[ety.union_all().difference(cov)], crs=32617)
fig, ax = plt.subplots(figsize=(10, 10))
ety.to_crs(4326).plot(ax=ax, color='lightgrey')
gaps.to_crs(4326).boundary.plot(ax=ax, color='red', linewidth=1.5)
plt.savefig('pred_results/Fire_Service_Analysis.png', dpi=100)
