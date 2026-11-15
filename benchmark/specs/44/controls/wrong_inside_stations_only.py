# Wrong: only the stations inside Etobicoke are buffered (2.5 km), so areas served by stations outside the
# district are reported as no-service.
import os, geopandas as gpd, matplotlib.pyplot as plt
os.makedirs("pred_results", exist_ok=True)
eto = gpd.read_file("dataset/etobicoke.geojson")
fs = gpd.read_file("dataset/fire_station.geojson")
e = eto.to_crs(26917); s = fs.to_crs(26917)
s = s[s.within(e.union_all())]          # only Etobicoke's own stations
cover = s.buffer(2500).union_all()
gap = gpd.GeoDataFrame(geometry=[e.union_all().difference(cover)], crs=26917).explode(index_parts=False)
gap.to_file("pred_results/no_service_area.geojson", driver="GeoJSON")
ax = eto.plot(color="#eee", edgecolor="k", figsize=(9, 9))
gap.to_crs(eto.crs).plot(ax=ax, color="red")
plt.savefig("pred_results/no_service_map.png")
