# Wrong: CRS error, ~2.5 km converted to degrees (0.0225) and buffered in lat/lon -> distorted ellipses.
import geopandas as gpd, matplotlib.pyplot as plt
e = gpd.read_file('dataset/etobicoke.geojson'); s = gpd.read_file('dataset/fire_station.geojson')
buf = gpd.GeoDataFrame(geometry=s.buffer(2500 / 111000), crs=s.crs)
ns = gpd.overlay(e, buf, how='difference')
ns.to_file('pred_results/no_service.geojson')
ax = e.plot(); ns.plot(ax=ax, color='red'); plt.savefig('pred_results/map.png')
