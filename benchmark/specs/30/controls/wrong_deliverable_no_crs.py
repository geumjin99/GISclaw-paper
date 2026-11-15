"""Wrong: the analysis is the Esri tutorial variant (correct_tutorial_broadway), but every geographic deliverable file is rewritten
without a CRS; geographic deliverables must carry a CRS.
"""
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
east = arterials[(arterials["ORD_STREET"] == "MADISON") & (arterials.distance(ch_geom) <= 3.5)]
west = arterials[arterials["ORD_STNAME"].isin(["10TH AVE E", "BROADWAY", "BROADWAY E"]) & arterials.intersects(ch_geom)]
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


# ---------------------------------------------------------------------------------------------
# NEGATIVE CONTROL: the analysis above is the Esri tutorial variant (correct_tutorial_broadway), but every geographic
# deliverable file is then written without a CRS. Expected: wrong.
def _write_without_crs(path):
    """Rewrite one file in place so that it carries no CRS. Returns a short description.
    .shp: drop the .prj sidecar. .tif/.tiff: rewrite without CRS (and drop .aux.xml, which can
    hold one). .gpkg: rewrite every spatial layer with crs=None. .geojson/.json: write projected
    coordinates (UTM if the data are lon/lat) without a 'crs' member - a lon/lat GeoJSON
    without 'crs' is WGS 84 by definition (RFC 7946), so dropping the member alone would not
    remove the CRS."""
    import os
    low = path.lower()
    stem, _ = os.path.splitext(path)
    if low.endswith(".shp"):
        d, b = os.path.split(stem)
        gone = []
        for n in os.listdir(d or "."):
            if n.lower() == (b + ".prj").lower():
                os.remove(os.path.join(d, n))
                gone.append(n)
        return f"removed {gone}"
    if low.endswith((".tif", ".tiff")):
        import rasterio
        with rasterio.open(path) as r:
            prof, data = r.profile.copy(), r.read()
            tags = r.tags()
        prof.pop("crs", None)
        tmp = stem + ".__nocrs__" + os.path.splitext(path)[1]
        with rasterio.open(tmp, "w", **prof) as w:
            w.write(data)
            if tags:
                w.update_tags(**tags)
        os.replace(tmp, path)
        if os.path.exists(path + ".aux.xml"):
            os.remove(path + ".aux.xml")
        return "raster rewritten without CRS"
    if low.endswith(".gpkg"):
        import geopandas as gpd
        import pyogrio
        tmp = stem + ".__nocrs__.gpkg"
        for lyr, gt in pyogrio.list_layers(path):
            g = gpd.read_file(path, layer=lyr)
            if gt is not None:
                g = g.set_crs(None, allow_override=True)
            g.to_file(tmp, layer=lyr, driver="GPKG")
        os.replace(tmp, path)
        return "gpkg rewritten without CRS"
    if low.endswith((".geojson", ".json")):
        import geopandas as gpd
        g = gpd.read_file(path)
        if g.crs is None or g.crs.is_geographic:
            if g.crs is None:
                g = g.set_crs(4326)
            g = g.to_crs(g.estimate_utm_crs())
        g = g.set_crs(None, allow_override=True)
        os.remove(path)
        g.to_file(path, driver="GeoJSON")
        return "GeoJSON rewritten in projected coordinates without 'crs'"
    return "not a geographic file"


def _strip_deliverable_crs(patterns):
    import fnmatch
    import os
    for n in sorted(os.listdir("pred_results")):
        low = n.lower()
        if low.endswith(('.shp', '.geojson', '.gpkg', '.json', '.tif', '.tiff')) and any(fnmatch.fnmatchcase(low, p.lower()) for p in patterns):
            if low.endswith(".json"):
                try:
                    import json
                    if json.load(open(os.path.join("pred_results", n))).get("type") is None:
                        continue
                except Exception:
                    continue
            print("no-CRS control:", n, _write_without_crs(os.path.join("pred_results", n)))


_strip_deliverable_crs(['*.geojson', '*.gpkg', '*.json', '*.shp'])
