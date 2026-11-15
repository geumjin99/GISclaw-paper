# Wrong: subset of stations, only MUN_NAME containing 'Etobicoke' (12 stations), 2.5 km radius.
import geopandas as gpd, matplotlib.pyplot as plt
e = gpd.read_file('dataset/etobicoke.geojson').to_crs(26917)
s = gpd.read_file('dataset/fire_station.geojson').to_crs(26917)
s = s[s.MUN_NAME.str.contains('Etobicoke', case=False, na=False)]
gap = gpd.GeoDataFrame(geometry=[e.union_all().difference(s.buffer(2500).union_all())], crs=26917)
gap.to_file('pred_results/no_service.geojson', driver='GeoJSON')
ax = e.plot(color='#eee', edgecolor='k'); gap.plot(ax=ax, color='red')
plt.savefig('pred_results/gaps.png')
