"""Reference solution for task 44: fire-station coverage gaps in Etobicoke.

Adapted from the reference script of GeoAnalystBench (Zhang et al. 2025, Apache-2.0).

Both layers are delivered in geographic coordinates (EPSG:4326), so a 2 500 m buffer must be
built in a metric CRS: NAD83 / UTM zone 17N (EPSG:26917). All 84 Toronto stations are buffered
(stations outside Etobicoke still serve its edges), the buffers are dissolved, and the
no-service area is Etobicoke minus the covered area. The 2 500 m distance is the one used by
the original solution (the instruction gives none).
Outputs: pred_results/fire_buffer.geojson, pred_results/no_service.geojson,
pred_results/fire_coverage_gaps.png
"""
import geopandas as gpd
import matplotlib.pyplot as plt

METRIC = "EPSG:26917"
DIST = 2500.0

etobicoke = gpd.read_file("dataset/etobicoke.geojson").to_crs(METRIC)
stations = gpd.read_file("dataset/fire_station.geojson").to_crs(METRIC)
print("stations:", len(stations), "in Etobicoke:", int(stations.within(etobicoke.union_all()).sum()))

fire_buffer = gpd.GeoDataFrame(stations[["NAME", "ADDRESS"]].copy(), geometry=stations.buffer(DIST, resolution=32),
                               crs=METRIC)
fire_buffer.to_file("pred_results/fire_buffer.geojson", driver="GeoJSON")

coverage = gpd.GeoDataFrame(geometry=[fire_buffer.union_all()], crs=METRIC)
no_service = gpd.overlay(etobicoke, coverage, how="difference")
no_service = no_service.explode(index_parts=False).reset_index(drop=True)
no_service["area_km2"] = no_service.area / 1e6
no_service = no_service[no_service.area_km2 > 1e-6].sort_values("area_km2", ascending=False)
no_service.to_file("pred_results/no_service.geojson", driver="GeoJSON")
total = etobicoke.area.sum() / 1e6
print(f"Etobicoke {total:.1f} km2, no service {no_service.area_km2.sum():.2f} km2 "
      f"({100 * no_service.area_km2.sum() / total:.1f} %) in {len(no_service)} pieces")

fig, ax = plt.subplots(figsize=(9, 9))
etobicoke.plot(ax=ax, color="whitesmoke", edgecolor="black")
fire_buffer.clip(etobicoke).plot(ax=ax, color="lightgreen", alpha=0.5)
no_service.plot(ax=ax, color="red")
stations.clip(etobicoke.buffer(DIST)).plot(ax=ax, color="darkred", markersize=15, marker="^")
ax.set_title(f"Etobicoke fire-station coverage ({DIST:.0f} m) and no-service areas (red)")
ax.set_axis_off()
plt.savefig("pred_results/fire_coverage_gaps.png", dpi=120, bbox_inches="tight")
