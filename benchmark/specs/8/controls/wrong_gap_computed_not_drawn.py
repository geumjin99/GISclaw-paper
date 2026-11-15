# Wrong: the correct gap (UTM 17N, 2.5 km, all stations) is computed, but the saved map shows only the
# district and the buffers - the coverage gap is never drawn.
import geopandas as gpd, matplotlib.pyplot as plt

ety = gpd.read_file('dataset/etobicoke.geojson').to_crs(32617)
st = gpd.read_file('dataset/fire_station.geojson').to_crs(32617)
cov = gpd.GeoDataFrame(geometry=[st.buffer(2500).union_all()], crs=32617)
gaps = gpd.overlay(ety, cov, how='difference')
fig, ax = plt.subplots(figsize=(10, 10))
ety.plot(ax=ax, color='lightgrey', edgecolor='black')
cov.plot(ax=ax, color='blue', alpha=0.3)
plt.savefig('pred_results/Fire_Service_Analysis.png', dpi=100)
