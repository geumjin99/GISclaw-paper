"""Reference solution for task 2: public transit access in Hamilton County, TN.

Overlay the bus-stop service area with three block-group demographic layers (population
density, households below poverty, households without a vehicle), each drawn with its own
semi-transparent colour ramp, and quantify the overlay (intersection / difference).

Note: the 'populationToPolygonSizeRating' column named in the task as "population density" is a
constant (2.191) in the supplied data, so it cannot show any pattern. Population density is
therefore derived as total population / land area (persons per km2).
"""
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CRS_M = "EPSG:32616"  # UTM 16N (metres) for areas


def main():
    service = gpd.read_file("dataset/BusServiceArea.geojson").to_crs(CRS_M)
    demo = gpd.read_file("dataset/HamiltonDemographics.geojson").to_crs(CRS_M)

    # --- three demographic factors per block group
    demo["area_km2"] = demo.geometry.area / 1e6
    rating = demo["populationToPolygonSizeRating"]
    print("populationToPolygonSizeRating unique values:", rating.unique()[:5])
    demo["pop_density"] = demo["populationtotals_TOTPOP_CY"] / demo["area_km2"]
    demo["poverty_hh"] = demo["households_ACSHHBPOV"].fillna(0)
    demo["no_vehicle_hh"] = demo["AtRisk_ACSOVEH0"].fillna(0)

    # --- overlay with the service area
    service_u = service.dissolve()[["geometry"]]
    served = gpd.overlay(demo, service_u, how="intersection")          # demographics inside
    unserved = gpd.overlay(demo, service_u, how="difference")          # demographics outside
    served["part_km2"] = served.geometry.area / 1e6
    demo["served_share"] = (served.groupby("OBJECTID")["part_km2"].sum()
                            .reindex(demo["OBJECTID"]).fillna(0).values / demo["area_km2"].values)

    # summary of the overlay: how much of each need indicator lies inside the service area
    summary = {}
    for c in ("populationtotals_TOTPOP_CY", "poverty_hh", "no_vehicle_hh"):
        inside = (demo[c] * demo["served_share"]).sum()  # area-weighted apportionment
        summary[c] = inside / demo[c].sum()
    print("share of county total inside the bus service area:", {k: round(v, 3) for k, v in summary.items()})

    # --- map: three transparent colour schemes + the service area
    fig, ax = plt.subplots(figsize=(12, 12))
    demo.boundary.plot(ax=ax, color="lightgrey", linewidth=0.3)
    layers = [("pop_density", "Greens", "Population density (persons/km$^2$)"),
              ("poverty_hh", "Reds", "Households below poverty"),
              ("no_vehicle_hh", "Blues", "Households without a vehicle")]
    handles = []
    for col, cmap, label in layers:
        v = demo[col]
        q = (v / v.quantile(0.95)).clip(0, 1)  # 0 = none, 1 = top 5 %; zeros stay transparent-light
        demo.assign(_q=q).plot(ax=ax, column="_q", cmap=cmap, alpha=0.35, linewidth=0)
        handles.append(Patch(color=plt.get_cmap(cmap)(0.8), alpha=0.6, label=label))
    service_u.boundary.plot(ax=ax, color="black", linewidth=1.8)
    handles.append(Patch(facecolor="none", edgecolor="black", label="Bus stop service area"))
    ax.legend(handles=handles, loc="lower right")
    ax.set_title("Bus service area and demographic need, Hamilton County, TN")
    ax.set_axis_off()
    plt.savefig("pred_results/transit_access.png", dpi=200, bbox_inches="tight")
    return demo, served, unserved


demo, served, unserved = main()
