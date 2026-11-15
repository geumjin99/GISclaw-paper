# Correct: NAD83(CSRS) / MTM zone 10 (EPSG:2952), 1.5 km service radius, unary_union + difference, shapefile.
import geopandas as gpd, matplotlib.pyplot as plt
e = gpd.read_file('dataset/etobicoke.geojson').to_crs(2952)
s = gpd.read_file('dataset/fire_station.geojson').to_crs(2952)
buf = s.copy(); buf['geometry'] = s.buffer(1500)
buf.to_file('pred_results/station_buffers.shp')
gap = e.geometry.union_all().difference(buf.geometry.union_all())
gpd.GeoDataFrame({'desc': ['no service']}, geometry=[gap], crs=2952).to_file('pred_results/gaps.shp')
ax = e.plot(color='white', edgecolor='k'); gpd.GeoSeries([gap], crs=2952).plot(ax=ax, color='red')
plt.savefig('pred_results/coverage.png')
