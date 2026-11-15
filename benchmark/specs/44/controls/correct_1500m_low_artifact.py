# Correct: 1.5 km analysis (UTM 17N, all stations) whose saved gap layer also carries a few tiny spurious
# pieces (here: three 10 m squares ~300 m from a station, < 0.01 % of the gap), the kind of debris an
# overlay can leave behind.
import geopandas as gpd, matplotlib.pyplot as plt
from shapely.geometry import box
from shapely.ops import unary_union
e = gpd.read_file('dataset/etobicoke.geojson').to_crs(32617)
s = gpd.read_file('dataset/fire_station.geojson').to_crs(32617)
E = e.union_all()
gap = E.difference(s.buffer(1500).union_all())
p = s[s.within(E)].geometry.iloc[0]
debris = unary_union([box(p.x + dx, p.y + 300, p.x + dx + 10, p.y + 310) for dx in (0, 40, 80)]).intersection(E)
gdf = gpd.GeoDataFrame(geometry=[gap.union(debris)], crs=32617).explode(index_parts=False)
gdf.to_file('pred_results/no_service_area.geojson', driver='GeoJSON')
ax = e.plot(color='#eee', edgecolor='k', figsize=(8, 8)); gdf.plot(ax=ax, color='red')
plt.savefig('pred_results/no_service_map.png')
