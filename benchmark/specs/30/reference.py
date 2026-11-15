"""Reference solution for task 30: bike routes for commuters to downtown Seattle (Esri case study
'Designate bike routes for commuting professionals'), open-source version.

Follows the Instruction: study neighbourhood Capitol Hill (L_HOOD = 'CAPITOL HILL'),
employment centres = Downtown Office Core 1/2 zoning, existing trails (SEGMENT_TY = 8),
candidate network = arterial streets (SEGMENT_TY = 1 and ARTERIAL_C = 1), three proposed
corridors as the Instruction names them:
  East    = "E Madison St in Capitol Hill and Madison St downtown": MADISON arterials within
            3.5 ft of Capitol Hill (the Esri model's rule) plus MADISON arterials that
            intersect the DOWNTOWN neighbourhood (L_HOOD = 'DOWNTOWN', i.e. Central Business
            District + First Hill). Together these form the continuous Madison corridor from
            the downtown office core to Capitol Hill.
  West    = 10TH AVE E only (the Esri model also adds BROADWAY / BROADWAY E; the task text
            does not, so they are left out - the tutorial variant is a positive control);
  Central = 23RD AVE, 23RD AVE E, 24TH AVE E, TURNER WAY E;
West and Central segments must touch Capitol Hill. Half-mile buffers, union, dissolve,
clip to Capitol Hill and coverage share. Network analysis (networkx on the arterial network)
checks that the proposed corridors connect Capitol Hill to the downtown office core."""
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point
from shapely.ops import nearest_points
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HALF_MILE_FT = 2640.0          # CRS: NAD83(HARN) / Washington North (US survey feet)
SQFT_PER_SQMI = 5280.0 ** 2

neighborhoods = gpd.read_file("dataset/Neighborhoods.shp")
streets = gpd.read_file("dataset/Streets.shp")
zoning = gpd.read_file("dataset/Zoning.shp")
assert neighborhoods.crs == streets.crs == zoning.crs

# 1-2. study area and employment centres; zoning joined to the neighbourhood (land-use mix)
capitol_hill = neighborhoods[neighborhoods["L_HOOD"] == "CAPITOL HILL"].dissolve(by="L_HOOD").reset_index()
ch_geom = capitol_hill.geometry.iloc[0]
ch_area_sqmi = ch_geom.area / SQFT_PER_SQMI
office_core = zoning[zoning["ZONELUT_DE"].isin(["Downtown Office Core 1", "Downtown Office Core 2"])]
ch_zoning = gpd.overlay(zoning[["ZONELUT_DE", "CLASS_DESC", "geometry"]], capitol_hill[["L_HOOD", "geometry"]],
                        how="intersection")
ch_zoning["area_sqmi"] = ch_zoning.area / SQFT_PER_SQMI
land_use_mix = ch_zoning.groupby("CLASS_DESC")["area_sqmi"].sum().sort_values(ascending=False)
print(f"Capitol Hill: {ch_area_sqmi:.2f} sq mi; land use:\n{(land_use_mix / ch_area_sqmi * 100).round(1)}")

# 3. classify streets: existing trails and arterial network
trails_ch = streets[(streets["SEGMENT_TY"] == 8) & streets.within(ch_geom)]
trail_length_ft = float(trails_ch["GIS_SEG_LE"].sum())
arterials = streets[(streets["SEGMENT_TY"] == 1) & (streets["ARTERIAL_C"] == 1)].copy()
print(f"existing trails in Capitol Hill: {len(trails_ch)} segments, {trail_length_ft:.0f} ft; arterials: {len(arterials)}")

# 4-6. proposed corridors (busy arterials that give direct routes, per the design)
downtown = neighborhoods[neighborhoods["L_HOOD"] == "DOWNTOWN"].union_all()
madison = arterials[arterials["ORD_STREET"] == "MADISON"]
east = madison[(madison.distance(ch_geom) <= 3.5) | madison.intersects(downtown)]
west = arterials[(arterials["ORD_STNAME"] == "10TH AVE E") & arterials.intersects(ch_geom)]
central = arterials[arterials["ORD_STNAME"].isin(["23RD AVE", "23RD AVE E", "24TH AVE E", "TURNER WAY E"])
                    & arterials.intersects(ch_geom)]
proposed_routes = pd.concat([east.assign(corridor="East"), west.assign(corridor="West"),
                             central.assign(corridor="Central")])
proposed_routes = gpd.GeoDataFrame(proposed_routes, geometry="geometry", crs=streets.crs)
route_lengths = proposed_routes.groupby("corridor").apply(lambda d: d.length.sum() / 5280.0)
print("proposed corridor length (mi):\n", route_lengths.round(2))

# network analysis: connectivity of each corridor to the downtown office core over arterials
G = nx.Graph()
for idx, geom in zip(arterials.index, arterials.geometry):
    for part in getattr(geom, "geoms", [geom]):
        c = list(part.coords)
        a, b = tuple(np.round(c[0], 0)), tuple(np.round(c[-1], 0))
        G.add_edge(a, b, length=part.length, idx=idx)
nodes = np.array(list(G.nodes))
def nearest_node(pt):
    return tuple(nodes[np.argmin((nodes[:, 0] - pt.x) ** 2 + (nodes[:, 1] - pt.y) ** 2)])
core_pt = office_core.union_all().centroid
core_node = nearest_node(core_pt)
dist_from_core = nx.single_source_dijkstra_path_length(G, core_node, weight="length")
connector_rows = []
for name, sub in proposed_routes.groupby("corridor"):
    ends = [nearest_node(Point(p)) for g in sub.geometry for part in getattr(g, "geoms", [g])
            for p in (part.coords[0], part.coords[-1])]
    reach = [(dist_from_core[n], n) for n in ends if n in dist_from_core]
    d, n = min(reach)
    path = nx.shortest_path(G, core_node, n, weight="length")
    connector_rows.append({"corridor": name, "network_dist_to_office_core_mi": d / 5280.0,
                           "connected": True, "path_nodes": len(path)})
connectivity = pd.DataFrame(connector_rows)
print(connectivity.round(2).to_string(index=False))

# 7. accessibility: half-mile coverage of Capitol Hill
buffers = [gpd.GeoSeries([d.geometry.union_all()], crs=streets.crs).buffer(HALF_MILE_FT)
           for _, d in proposed_routes.groupby("corridor")]
coverage = gpd.GeoDataFrame(geometry=[pd.concat(buffers).union_all().intersection(ch_geom)], crs=streets.crs)
coverage_share = float(coverage.area.iloc[0] / ch_geom.area)
print(f"share of Capitol Hill within a half mile of the proposed routes: {coverage_share:.1%}")

summary = pd.DataFrame({
    "metric": ["capitol_hill_area_sqmi", "existing_trail_length_ft", "arterial_segments",
               "east_route_mi", "west_route_mi", "central_route_mi", "half_mile_coverage_share"],
    "value": [ch_area_sqmi, trail_length_ft, len(arterials), route_lengths.get("East"),
              route_lengths.get("West"), route_lengths.get("Central"), coverage_share]})
summary.to_csv("pred_results/bike_route_summary.csv", index=False)
# GeoPackage keeps the State Plane (US ft) CRS; GeoJSON writers drop non-WGS84 CRSs
proposed_routes[["ORD_STNAME", "corridor", "geometry"]].to_file("pred_results/proposed_bike_routes.gpkg")
coverage.assign(share=coverage_share).to_file("pred_results/half_mile_coverage.gpkg")

# 9. map
fig, ax = plt.subplots(figsize=(9, 11))
xmin, ymin, xmax, ymax = ch_geom.buffer(6000).bounds
zoning.cx[xmin:xmax, ymin:ymax].plot(ax=ax, color="#eeeeee", edgecolor="white", linewidth=0.2)
office_core.plot(ax=ax, color="#c9a227", label="Downtown office core")
coverage.plot(ax=ax, color="#9ecae1", alpha=0.6)
capitol_hill.boundary.plot(ax=ax, color="black", linewidth=1.2)
arterials.cx[xmin:xmax, ymin:ymax].plot(ax=ax, color="#bbbbbb", linewidth=0.6)
trails_ch.plot(ax=ax, color="green", linewidth=1.5)
colors = {"East": "#d62728", "West": "#9467bd", "Central": "#ff7f0e"}
for name, sub in proposed_routes.groupby("corridor"):
    sub.plot(ax=ax, color=colors[name], linewidth=3, label=f"{name} ({route_lengths[name]:.1f} mi)")
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_axis_off()
ax.legend(loc="lower left")
ax.set_title(f"Proposed protected bike routes, Capitol Hill\n{coverage_share:.0%} of the neighbourhood "
             f"within 1/2 mile (blue); green = existing trails")
fig.savefig("pred_results/bike_routes.png", dpi=110, bbox_inches="tight")
