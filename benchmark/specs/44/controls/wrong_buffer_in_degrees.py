# Wrong: CRS error, buffer(2500) applied in EPSG:4326 (degrees) -> every point 'covered', gap empty.
import geopandas as gpd, matplotlib.pyplot as plt
e = gpd.read_file('dataset/etobicoke.geojson'); s = gpd.read_file('dataset/fire_station.geojson')
buf = gpd.GeoDataFrame(geometry=s.buffer(2500), crs=s.crs)
buf.to_file('pred_results/fire_buffer.geojson')
ns = gpd.overlay(e, buf, how='difference')
ns.to_file('pred_results/no_service.geojson')
ax = e.plot(); ns.plot(ax=ax, color='red'); plt.savefig('pred_results/map.png')
