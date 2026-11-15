# Wrong: subset of stations, only the 22 stations inside the district, 1 km radius. The gap is close to the
# all-station 1 km gap (IoU ~0.98) but keeps ~2 % of areas served by stations outside the district, which
# serve its edges.
import geopandas as gpd, matplotlib.pyplot as plt
e = gpd.read_file('dataset/etobicoke.geojson').to_crs(26917)
s = gpd.read_file('dataset/fire_station.geojson').to_crs(26917)
s = s[s.within(e.union_all())]
gap = gpd.GeoDataFrame(geometry=[e.union_all().difference(s.buffer(1000).union_all())], crs=26917)
gap.to_file('pred_results/no_service.geojson', driver='GeoJSON')
ax = e.plot(color='#eee', edgecolor='k'); gap.plot(ax=ax, color='red')
plt.savefig('pred_results/gaps.png')
