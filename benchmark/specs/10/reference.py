"""Reference solution for task 10: potential deforestation impact of the planned road BR 421, Rondonia.

  * all layers in South America Albers Equal Area (ESRI:102033, the CRS of planned_road.geojson;
    metres, equal-area);
  * 5.5 km buffer around the planned road (the distance within which deforestation concentrates
    along existing roads, see task 9);
  * risky area = buffer minus the already deforested area (forest that the road would newly
    expose); protected forest at risk = risky area intersected with the protected areas;
  * map: Rondonia boundary (Natural Earth admin-1, offline via cartopy), protected forest,
    deforested area, existing roads, planned road, its buffer and the risky area.
"""
import numpy as np
import geopandas as gpd
import shapely
from shapely.geometry import box
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

CRS_M = "ESRI:102033"
DIST = 5500.0


def rondonia_boundary():
    try:
        import cartopy.io.shapereader as shpreader
        p = shpreader.natural_earth(resolution="10m", category="cultural", name="admin_1_states_provinces_lakes")
        st = gpd.read_file(p)
        st = st[(st["admin"] == "Brazil") & (st["name"].str.contains("Rond"))]
        return st[["name", "geometry"]].to_crs(CRS_M)
    except Exception as e:  # boundary is context only
        print("no Rondonia boundary available:", e)
        return None


def main():
    roads = gpd.read_file("dataset/roads.geojson").to_crs(CRS_M)
    deforested = gpd.read_file("dataset/deforestedArea.geojson").to_crs(CRS_M)
    protected = gpd.read_file("dataset/protectedForest.geojson").to_crs(CRS_M)
    planned = gpd.read_file("dataset/planned_road.geojson").to_crs(CRS_M)
    state = rondonia_boundary()

    buffer = gpd.GeoDataFrame({"name": ["BR 421 5.5 km buffer"]},
                              geometry=[planned.geometry.union_all().buffer(DIST)], crs=CRS_M)
    buf = buffer.geometry.iloc[0]
    # clip the (very large, partly invalid) deforestation polygon to the buffer before overlaying
    defo = shapely.make_valid(shapely.clip_by_rect(deforested.geometry.iloc[0], *buf.bounds))
    defo_in_buf = defo.intersection(buf)
    risky = buf.difference(defo_in_buf)
    risky_area = gpd.GeoDataFrame({"name": ["risky area"]}, geometry=[risky], crs=CRS_M)
    prot = shapely.make_valid(protected.geometry.union_all())
    prot_at_risk = gpd.GeoDataFrame({"name": ["protected forest at risk"]},
                                    geometry=[risky.intersection(prot)], crs=CRS_M)
    hit = protected[protected.intersects(risky)]
    print(f"buffer {buf.area / 1e6:.0f} km2, already deforested {defo_in_buf.area / 1e6:.0f} km2 "
          f"({defo_in_buf.area / buf.area:.1%}), risky (still forested) {risky.area / 1e6:.0f} km2, "
          f"of which protected {prot_at_risk.area.iloc[0] / 1e6:.0f} km2 in {len(hit)} areas: {hit['Name'].tolist()}")

    fig, ax = plt.subplots(figsize=(12, 11))
    if state is not None:
        state.boundary.plot(ax=ax, color="black", linewidth=1.2)
    protected.plot(ax=ax, color="#a6d96a", alpha=0.8)
    deforested.plot(ax=ax, color="#d8b365", alpha=0.6)
    roads.plot(ax=ax, color="grey", linewidth=0.2)
    risky_area.plot(ax=ax, color="red", alpha=0.6)
    prot_at_risk.plot(ax=ax, color="darkred")
    buffer.boundary.plot(ax=ax, color="red", linewidth=0.6)
    planned.plot(ax=ax, color="blue", linewidth=1.5)
    ax.legend(handles=[Line2D([], [], color="black", label="Rondonia boundary"),
                       Patch(color="#a6d96a", label="Protected forest"),
                       Patch(color="#d8b365", label="Deforested area"),
                       Line2D([], [], color="grey", label="Existing roads"),
                       Line2D([], [], color="blue", label="Planned road (BR 421)"),
                       Patch(color="red", alpha=0.6, label="At risk: forest within 5.5 km of planned road"),
                       Patch(color="darkred", label="Protected forest at risk")], loc="lower left", fontsize=8)
    ax.set_title("Predicted deforestation risk from the planned road BR 421, Rondonia")
    ax.set_axis_off()
    plt.savefig("pred_results/predictedRiskyArea.png", dpi=150, bbox_inches="tight")
    return risky_area, prot_at_risk, buffer


risky_area, prot_at_risk, buffer = main()
