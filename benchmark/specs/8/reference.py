"""Reference solution for task 8: fire station coverage gaps in Etobicoke, Toronto.

The file is dataset/fire_station.geojson (the task text says fire_stations.geojson).
All 84 Toronto fire stations are buffered - stations just outside Etobicoke also cover its
edges - by 2.5 km (approx. the distance an engine covers in the 4-minute NFPA 1710 travel time
in an urban grid; the distance used by the benchmark's script) in UTM 17N (metres). The
coverage gap is Etobicoke minus the dissolved buffers.
"""
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

CRS_M = "EPSG:32617"
DIST = 2500.0


def main():
    stations = gpd.read_file("dataset/fire_station.geojson").to_crs(CRS_M)
    etobicoke = gpd.read_file("dataset/etobicoke.geojson").to_crs(CRS_M)
    coverage = gpd.GeoDataFrame(geometry=[stations.buffer(DIST).union_all()], crs=CRS_M)
    area = etobicoke.dissolve()[["geometry"]]
    no_service = gpd.overlay(area, coverage, how="difference")
    gap_km2 = no_service.area.sum() / 1e6
    print(f"Etobicoke {area.area.sum() / 1e6:.1f} km2; uncovered by {DIST / 1000:.1f} km buffers: "
          f"{gap_km2:.2f} km2 ({gap_km2 / (area.area.sum() / 1e6):.1%})")

    fig, ax = plt.subplots(figsize=(10, 10))
    etobicoke.plot(ax=ax, color="whitesmoke", edgecolor="grey")
    gpd.overlay(area, coverage, how="intersection").plot(ax=ax, color="lightsteelblue", alpha=0.7)
    no_service.plot(ax=ax, color="crimson")
    stations[stations.intersects(area.buffer(DIST).union_all())].plot(ax=ax, color="black", markersize=12)
    ax.legend(handles=[Patch(color="lightsteelblue", label=f"Within {DIST / 1000:.1f} km of a fire station"),
                       Patch(color="crimson", label=f"Coverage gap ({gap_km2:.2f} km$^2$)"),
                       Line2D([], [], marker="o", color="black", linestyle="", label="Fire station")],
              loc="lower left")
    xmin, ymin, xmax, ymax = area.total_bounds
    ax.set_xlim(xmin - 1000, xmax + 1000)
    ax.set_ylim(ymin - 1000, ymax + 1000)
    ax.set_title("Fire station service coverage and gaps in Etobicoke")
    ax.set_axis_off()
    plt.savefig("pred_results/Fire_Service_Analysis.png", dpi=150, bbox_inches="tight")
    return no_service, coverage


no_service, coverage = main()
