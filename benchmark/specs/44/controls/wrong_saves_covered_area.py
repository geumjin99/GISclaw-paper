# Wrong: saves the covered (intersection) area instead of the no-service area.
import geopandas as gpd, matplotlib.pyplot as plt
e = gpd.read_file('dataset/etobicoke.geojson').to_crs(32617); s = gpd.read_file('dataset/fire_station.geojson').to_crs(32617)
buf = gpd.GeoDataFrame(geometry=s.buffer(2500), crs=32617)
cov = gpd.overlay(e, buf, how='intersection')
cov.to_file('pred_results/service_area.geojson')
ax = e.plot(color='red'); cov.plot(ax=ax, color='green'); plt.savefig('pred_results/map.png')
