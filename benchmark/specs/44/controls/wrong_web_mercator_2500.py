# Wrong: the 2 500 m buffer is built in Web Mercator (EPSG:3857), which is only ~1.8 km on the ground at
# this latitude.
import os, geopandas as gpd, matplotlib.pyplot as plt
os.makedirs("pred_results", exist_ok=True)
eto = gpd.read_file("dataset/etobicoke.geojson")
fs = gpd.read_file("dataset/fire_station.geojson")
# "metric" CRS chosen as Web Mercator: 2500 map units = only ~1.8 km on the ground at 43.7 N
e = eto.to_crs(3857); s = fs.to_crs(3857)
cover = s.buffer(2500).union_all()
gap = gpd.GeoDataFrame(geometry=[e.union_all().difference(cover)], crs=3857).explode(index_parts=False)
gap.to_file("pred_results/no_service_area.geojson", driver="GeoJSON")
ax = eto.plot(color="#eee", edgecolor="k", figsize=(9, 9))
gap.to_crs(eto.crs).plot(ax=ax, color="red")
plt.savefig("pred_results/no_service_map.png")
